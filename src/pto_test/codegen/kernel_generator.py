"""
Kernel code generator for PTO testing framework.

Converts PyPTO Programs to CCE C++ kernel source files using
the PassManager and CceCodegen pipeline.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Add pypto to path
_FRAMEWORK_ROOT = Path(__file__).parent.parent.parent.parent
_PYPTO_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "pypto" / "python"

if _PYPTO_ROOT.exists() and str(_PYPTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYPTO_ROOT))

if TYPE_CHECKING:
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.pypto_core import ir as core_ir


class KernelGenerator:
    """Generates CCE C++ kernel source files from PyPTO Programs.

    Pipeline: ir.Program -> PassManager -> CceCodegen -> C++ source files

    Each function in the Program becomes a separate .cpp kernel file
    placed in the output directory under an aiv/ subdirectory.

    Example:
        from pypto.ir.pass_manager import OptimizationStrategy

        generator = KernelGenerator(strategy=OptimizationStrategy.PTOAS)
        kernel_configs = generator.generate(program, output_dir)
        # Returns: [{"func_id": 0, "source": "path/to/aiv/func.cpp", "core_type": "aiv"}, ...]
    """

    # Standard kernel header template
    KERNEL_HEADER = """#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

"""

    def __init__(
        self,
        strategy: Optional["OptimizationStrategy"] = None,
        core_type: str = "aiv",
    ):
        """Initialize kernel generator.

        Args:
            strategy: Optimization strategy for pass pipeline.
                      If None, uses OptimizationStrategy.Default.
            core_type: Target core type for kernels (default: "aiv").
        """
        # Import here to avoid circular imports and allow lazy loading
        from pypto.ir.pass_manager import OptimizationStrategy

        if strategy is None:
            strategy = OptimizationStrategy.Default

        self.strategy = strategy
        self.core_type = core_type
        self._codegen = None
        self._pass_manager = None

    def _get_codegen(self):
        """Lazily initialize CceCodegen."""
        if self._codegen is None:
            from pypto.pypto_core import codegen
            self._codegen = codegen.CceCodegen()
        return self._codegen

    def _get_pass_manager(self):
        """Lazily initialize PassManager."""
        if self._pass_manager is None:
            from pypto.ir.pass_manager import PassManager
            self._pass_manager = PassManager.get_strategy(self.strategy)
        return self._pass_manager

    @staticmethod
    def _add_extern_c(code: str) -> str:
        """Add 'extern "C"' to function declarations.

        PyPTO's CceCodegen generates:
            __aicore__ __attribute__((always_inline)) void func_name(...)

        This method transforms it to:
            extern "C" __aicore__ __attribute__((always_inline)) void func_name(...)

        Args:
            code: Generated C++ code from CceCodegen

        Returns:
            Code with extern "C" added to function declarations
        """
        # Pattern: Match __aicore__ at the start of a line (possibly with leading whitespace)
        # and prepend 'extern "C" ' to it
        pattern = r'^(\s*)(__aicore__\s+__attribute__\(\(always_inline\)\))'
        replacement = r'\1extern "C" \2'
        return re.sub(pattern, replacement, code, flags=re.MULTILINE)

    def generate(
        self,
        program: Any,
        output_dir: Path,
        dump_passes: bool = False,
    ) -> List[Dict[str, Any]]:
        """Generate CCE C++ kernel files from a PyPTO Program.

        Args:
            program: PyPTO Program (from @pl.program decorator or ir.Program).
            output_dir: Directory to write kernel files into.
                        Creates aiv/ (or core_type/) subdirectory.
            dump_passes: If True, dump intermediate IR after each pass.

        Returns:
            List of kernel config dicts compatible with simpler's kernel_config.py:
            [{"func_id": 0, "source": "path/to/kernel.cpp", "core_type": "aiv"}, ...]
        """
        output_dir = Path(output_dir)

        # Run pass pipeline
        pm = self._get_pass_manager()
        if dump_passes:
            passes_dir = output_dir / "pass_dump"  # Changed to singular
            passes_dir.mkdir(parents=True, exist_ok=True)
            transformed = pm.run_passes(program, dump_ir=True, output_dir=str(passes_dir))
        else:
            transformed = pm.run_passes(program)

        # Create core type subdirectory
        kernel_dir = output_dir / self.core_type
        kernel_dir.mkdir(parents=True, exist_ok=True)

        # Generate code for each function
        codegen = self._get_codegen()
        kernels = []

        for func_id, (func_name, func) in enumerate(transformed.functions.items()):
            # Generate a readable function name
            # func_name is a GlobalVar object - extract the actual name
            readable_name = None

            # Try multiple ways to extract the function name
            if hasattr(func_name, 'name_hint'):
                readable_name = func_name.name_hint
            if not readable_name and hasattr(func_name, '__name__'):
                readable_name = func_name.__name__
            if not readable_name and hasattr(func_name, 'name'):
                readable_name = func_name.name

            # Try to extract from the function object itself
            if not readable_name and hasattr(func, 'name'):
                readable_name = func.name
            if not readable_name and hasattr(func, 'name_hint'):
                readable_name = func.name_hint

            # Generate C++ code first - it might contain the function name
            code = codegen.Generate(func)

            # If still no name, try to extract from generated code
            # Look for pattern like "void function_name(" or "__aicore__ void function_name("
            if not readable_name:
                import re
                # Match function definition: void name(...) or __aicore__ ... void name(...)
                match = re.search(r'void\s+(\w+)\s*\(', code)
                if match:
                    readable_name = match.group(1)

            # Final fallback: use func_id or memory address
            if not readable_name:
                func_str = str(func_name)
                # Extract hex address from string like "<...object at 0xfffe612bc3d0>"
                addr_match = re.search(r'0x[0-9a-fA-F]+', func_str)
                if addr_match:
                    addr = addr_match.group(0)
                    readable_name = f"func_{addr}"
                else:
                    readable_name = f"func_{func_id}"

            # Add extern "C" to function declarations
            code = self._add_extern_c(code)

            # Add header to generated code
            code_with_header = self.KERNEL_HEADER + code

            # Write to file
            source_path = kernel_dir / f"{readable_name}.cpp"
            source_path.write_text(code_with_header)

            kernels.append({
                "func_id": func_id,
                "source": str(source_path),
                "core_type": self.core_type,
                "function_name": readable_name,
            })

        return kernels

    def generate_single(
        self,
        program: Any,
        func_name: Optional[str] = None,
    ) -> str:
        """Generate C++ code for a single function without writing to file.

        Args:
            program: PyPTO Program.
            func_name: Name of function to generate. If None and program has
                      exactly one function, uses that function.

        Returns:
            Generated C++ code as string.

        Raises:
            ValueError: If func_name is None and program has multiple functions.
            KeyError: If func_name not found in program.
        """
        pm = self._get_pass_manager()
        transformed = pm.run_passes(program)

        if func_name is None:
            if len(transformed.functions) == 1:
                func_name = list(transformed.functions.keys())[0]
            else:
                raise ValueError(
                    f"Program has {len(transformed.functions)} functions. "
                    "Specify func_name to select one."
                )

        if func_name not in transformed.functions:
            raise KeyError(
                f"Function '{func_name}' not found. "
                f"Available: {list(transformed.functions.keys())}"
            )

        func = transformed.functions[func_name]
        codegen = self._get_codegen()
        code = codegen.Generate(func)

        # Add extern "C" to function declarations
        code = self._add_extern_c(code)

        return self.KERNEL_HEADER + code
