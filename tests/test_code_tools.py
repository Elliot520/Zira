"""project_overview and code_outline."""

from __future__ import annotations

import pytest

from app.tools.code_tools import CodeOutlineTool, ProjectOverviewTool, outline
from app.tools.filesystem import FileAccessPolicy


@pytest.fixture
def android(tmp_path):
    root = tmp_path / "shop"
    (root / "app" / "src" / "main" / "java" / "com" / "x").mkdir(parents=True)
    (root / "app" / "build.gradle.kts").write_text(
        'android { defaultConfig { applicationId = "com.x.shop" } }\ndependencies { implementation("com.squareup.retrofit2:retrofit:2.9.0") }\n')
    (root / "app" / "src" / "main" / "AndroidManifest.xml").write_text("<manifest/>")
    (root / "app" / "src" / "main" / "java" / "com" / "x" / "Api.kt").write_text(
        "import retrofit2.http.*\ninterface Api {\n    @POST(\"user/login\")\n    fun login(@Body u: User): Call<User>\n    @GET(\"user/list\")\n    fun list(): Call<List<User>>\n}\n"
        "data class User(val id: Int = 0)\n")
    (root / "README.md").write_text("# Shop\nAn online shop app.\n")
    (root / "API_DOCS.md").write_text("| POST | `user/login` |\n")
    (root / "keystore.properties").write_text("storePassword=hunter2hunter2\n")
    return root


@pytest.fixture
def policy(android):
    return FileAccessPolicy([android])


async def test_overview_describes_stack_layout_and_api_code(policy, android):
    out = (await ProjectOverviewTool(policy).execute(path=str(android))).output
    assert "Android app" in out and "Gradle" in out
    assert "applicationId=com.x.shop" in out and "retrofit" in out
    assert "Retrofit endpoints: 2" in out
    assert "API endpoints (" in out and "POST /user/login" in out and "GET /user/list" in out
    assert "API_DOCS.md" in out and "README start" in out and "An online shop app" in out
    assert "app (" in out  # top-level layout
    assert "hunter2" not in out and "keystore" not in out


async def test_overview_rejects_bad_paths(policy):
    assert not (await ProjectOverviewTool(policy).execute(path="/etc")).ok
    assert not (await ProjectOverviewTool(policy).execute()).ok


async def test_overview_of_a_node_project(tmp_path):
    root = tmp_path / "web"
    root.mkdir()
    (root / "package.json").write_text('{"name": "web", "scripts": {"start": "node ."}, "dependencies": {"express": "^4"}}')
    (root / "server.js").write_text("const app = require('express')();\napp.get('/health', (req, res) => res.send('ok'));\n")
    out = (await ProjectOverviewTool(FileAccessPolicy([root])).execute(path=str(root))).output
    assert "Node.js" in out and "Express routes: 1" in out and "express" in out


KOTLIN = """package x
class App : Base() {
    fun hello() = "hi"
    private suspend fun load(): Int = 1
    companion object { const val X = 1 }
}
data class User(val id: Int)
interface Api { fun list(): List<User> }
object Util
"""

PYTHON = """import os
class Service:
    def run(self):
        pass
    async def start(self):
        pass
def helper():
    pass
"""

JS = """export async function fetchUsers() {}
const add = (a, b) => a + b;
export const load = async () => {};
class Store {}
export interface Props {}
"""

GO = """package main
type Server struct{}
func (s *Server) Handle() {}
func main() {}
"""

JAVA = """public class Main {
    public static void main(String[] args) {
    }
    private int compute(int x) throws Exception {
        return x;
    }
}
"""


@pytest.mark.parametrize(
    "suffix,source,expected",
    [
        (".kt", KOTLIN, {"class App", "fun hello", "fun load", "class User", "interface Api", "object Util"}),
        (".py", PYTHON, {"class Service", "def run", "async def start", "def helper"}),
        (".ts", JS, {"function fetchUsers", "class Store", "interface Props"}),
        (".go", GO, {"type Server", "func Handle", "func main"}),
        (".java", JAVA, {"class Main", "method main", "method compute"}),
    ],
)
def test_outline_recognises_declarations(suffix, source, expected):
    found = {f"{kind} {name}" for _, kind, name, _ in outline(source, suffix)}
    assert expected <= found


def test_outline_handles_arrow_functions_and_ignores_comments():
    names = {n for _, _, n, _ in outline("// class Fake\nconst add = (a, b) => a + b;\n", ".js")}
    assert names == {"add"}


def test_outline_of_unknown_language_is_empty():
    assert outline("whatever", ".xyz") == []


async def test_code_outline_file_and_folder(policy, android):
    tool = CodeOutlineTool(policy)
    file_out = (await tool.execute(path=str(android / "app/src/main/java/com/x/Api.kt"))).output
    assert "L2 interface Api" in file_out and "fun login" in file_out and "class User" in file_out
    folder = (await tool.execute(path=str(android))).output
    assert "Api.kt" in folder and "decl" in folder


async def test_code_outline_refuses_secrets_and_outside(policy, android):
    tool = CodeOutlineTool(policy)
    assert not (await tool.execute(path=str(android / "keystore.properties"))).ok
    assert not (await tool.execute(path="/etc/hosts")).ok


def test_symbol_list_is_capped():
    text = "\n".join(f"def f{i}(): pass" for i in range(500))
    assert len(outline(text, ".py")) == 100
