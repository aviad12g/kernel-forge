# Static Policy Checks

OpenKernelForge runs lightweight AST-based candidate checks before correctness verification. The implementation is in `openkernelforge/harness/policy.py`; template-copy checks are in `openkernelforge/harness/template_preservation.py`.

## Base Candidate Policy

| Check | Detection method | Example rejected pattern | Purpose |
| --- | --- | --- | --- |
| Python syntax | `ast.parse` | invalid Python source | reject unimportable candidates before local loading |
| Candidate entry point (`missing_forward`) | module-level `forward` or `ModelNew.forward` lookup | no supported `forward` method | enforce candidate contract |
| Suspicious task/reference imports (`imports_reference_or_task_module`) | `Import` and `ImportFrom` nodes | `from openkernelforge.tasks...`, names containing `reference` or `ref_` | prevent direct access to task references |
| Import allowlist (`disallowed_import:*`) | exact module allowlist when torch fallback is disabled | `import os`, `from torch.utils.cpp_extension import load` | prevent filesystem/process/native-extension escape paths |
| Import alias allowlist (`disallowed_import_alias:*`) | canonical aliases for Torch and Triton modules in strict mode | `import torch as hidden` | prevent alias-based bypasses of qualified-call checks |
| Direct Torch symbol import (`disallowed_from_torch_import:*`) | `ImportFrom` name allowlist in strict mode | `from torch import relu` | prevent aliasing high-level compute around qualified-call detection |
| Unsafe calls (`unsafe_call:*`) | full-module call scan | `open(...)`, `eval(...)`, `subprocess.run(...)`, `os.system(...)` | reject common side-effect and code-execution paths |
| Import-time calls (`import_time_call:*`) | scan module and class bodies plus decorators and function defaults while skipping function bodies | `torch.set_default_dtype(...)`, class-level tensor allocation, eager `.cuda().eval()` construction | reject side effects that execute before verification; standard Triton configuration and JIT/autotune decorators are allowed |
| Suspicious reference calls (`calls_suspicious_function:*`) | calls inside `forward` | `reference(...)`, `ref_forward(...)`, `torch_forward(...)` | prevent routing through the reference |
| PyTorch fallback when disabled (`obvious_torch_fallback:*`) | calls, tensor arithmetic, data attributes/subscripts, and non-metadata tensor methods in `forward` and reachable Python helpers | assigned `torch.relu`, `F.conv2d`, `alias.square()`, `x.T`, direct tensor arithmetic | reject high-level fallback hidden behind aliases or helper functions |
| Indirect module call (`calls_module_from_forward:*`) | `self.*(...)` calls in `ModelNew.forward` | `return self.conv(x)` | prevent wrapping the original PyTorch module as the candidate |
| Triton launch requirement (`missing_triton_kernel_launch`) | match a JIT-decorated kernel definition to an indexed launch reachable from the entry point | import Triton or place a launch in an unused helper | require actual Triton execution in strict KernelBench mode |
| Torch fallback allowed warning | fallback detected with `allow_torch_fallback=true` | `return x + y` | preserve visibility in explicit development/smoke mode |

The strict policy uses version `ast-v5`. It allows tensor allocation and shape/stride inspection in wrappers, but rejects high-level Torch compute calls in the entry point and reachable Python helpers. Version 4 added scans of class bodies, decorators, and function defaults for import-time execution. Version 5 also rejects obvious in-place tensor-compute fallbacks such as `add_`, `copy_`, and `scatter_`. Triton usage requires both a JIT-decorated function and a matching reachable indexed launch; arbitrary or unreachable subscript calls do not count as Triton.

## Execution Boundary

Policy checking is a guardrail, not an operating-system sandbox. The historical module `openkernelforge.harness.sandbox` performs a trusted in-process import after policy checks and cleans up failed imports, but it does not isolate filesystem, network, process, or GPU access. Public evaluation of untrusted candidates should run each candidate in a disposable worker or container with an external timeout. The paper does not claim security isolation.

## Template-Preservation Policy

Template-guided checks are separate from the base policy and are used only when configured.

| Check | Detection method | Example rejected pattern | Purpose |
| --- | --- | --- | --- |
| Missing `forward` | AST function lookup | candidate lacks wrapper | ensure executable wrapper |
| Missing Triton JIT marker | source text contains `@triton.jit` | copied template drops kernel | preserve kernel implementation |
| Missing `BLOCK_SIZE` | source text | no block-size meta-parameter | preserve tunable template structure |
| Kernel launch count changed | regex count of `kernel[grid](...)` | launch removed or duplicated | detect wrapper/fallback rewrites |
| Try/except wrapper | AST `Try` node | fallback in `except Exception` | detect hidden fallback path |
| Fallback/reference tokens | source text and forward AST | strings such as `fallback`, `reference`, `torch_forward` | detect reference-style escape hatches |
| Forbidden torch ops in `forward` | AST call scan | `torch.relu`, `torch.sigmoid`, `torch.matmul` | keep computation in Triton kernel |
| Low preservation score | heuristic score below threshold | many structural changes | reject degraded template copies |

These checks are heuristic guardrails. Passing policy does not prove semantic correctness; correctness verification remains mandatory.
