from typing import Dict, Any
from ..base import Baker
# Avoid circular types
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..base import CompilerContext

class MeshBaker(Baker):
    """
    Bakes the Shell Mesh (Vertex/Index buffers).
    """
    def name(self) -> str:
        return "mesh"

    async def bake(self, ctx: 'CompilerContext') -> Dict[str, bytes]:
        await ctx.ensure_shell_mesh()
        return {
            "shell_mesh": ctx.shell_mesh
        }
