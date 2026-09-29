"""API scanner + Postman builder, on small projects that reproduce real-world patterns."""

from __future__ import annotations

import json

import pytest

from app.tools.api_scanner import ModelIndex, parse_retrofit_interface, scan_project, split_top_level, strip_comments
from app.tools.filesystem import FileAccessPolicy
from app.tools.postman import GeneratePostmanCollectionTool, build_collection, display_name, slugify


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


API_KT = '''package com.x.network
import retrofit2.http.*
interface RetrofitInterface {
     @POST("category/searchBrand")
     fun searchBrand(@Body c: CommonCategory): Call<R>

     @POST
     fun createUser(@Url url : String, @Body user: User): Call<UserResponse>

     @POST
     fun list(@Url url : String, @Header("Authorization") authHeader: String, @Body data: Map<String,String>): Call<R>

     @POST
     fun getGeneric(@Url url : String, @Body request: ProductListRequest, @Header("Authorization") authHeader: String) : Call<R>

     @GET
     fun getGeneric(@Url url : String, @Header("Authorization") authHeader: String) : Call<R>

     @POST
     fun getGeneric(@Url url : String, @Header("Authorization") authHeader: String, @Body request: HashMap<String,String>) : Call<R>

     @Multipart
     @POST
     fun upload(
         @Url url : String,
         @Part("imageid") id: RequestBody?,
         @Part image: MultipartBody.Part?
     ): Call<R>

     @GET
     fun ext(@Url url : String): Call<R>

     @POST
     fun neverUsed(@Url url : String, @Body b: User): Call<R>

     @POST
     fun sendOtp(@Url url : String, @Body data: Map<String,String>) : Call<R>

     @POST
     fun addUpdate(@Url url : String, @Body a: Address): Call<R>

     @POST
     fun search(@Url url : String, @Body c: CommonCategory): Call<R>

     @GET("items/{id}")
     fun item(@Path("id") id: Int, @Query("expand") expand: String): Call<R>
}
'''

CONSTANTS_KT = '''package com.x.utils
class AppConstants { companion object {
    val URL_CREATE_USER: String = "user/createUser"
    val URL_LIST: String = "cart/list"
    val URL_SEND_OTP: String = "user/sendOtp"
    val URL_VERIFY: String = "user/verifyOtp"
    val URL_UPLOAD : String = "uploader/uploadImage"
    val URL_GENERIC : String = "data/generic"
    val URL_NEVER_CALLED : String = "misc/neverCalled"
    val MIME : String = "image/jpeg"
    const val QUERY_TYPE_FROM_CONST : String = "fromconst"
}}
'''

MODELS_KT = '''package com.x.model
import com.google.gson.annotations.SerializedName
enum class Role { Admin, Member }
class User {
    var user_id: Int = 0
    var email: String? = null
    var role: Role? = null
    var kind: Role = Role.Member
    var tags: List<String>? = null
    @SerializedName("allow_radius_in_meter")
    var radius: Int? = 2000
    var name: String = "guest"
    @Transient var cache: String = "x"
    val computed: String get() = "no"
    val lazyOne by lazy { 1 }
}
class CommonCategory {
    var id: Int = 0
    var title: String? = null
    var search: String = ""
}
data class ProductListRequest(
    val offset: Int,
    val limit: Int,
    val screentype: String = "home",
    var query_type : String = "",
    var response_key : String = "",
    var latitude : Double = 0.0,
    var stock : Boolean = false,
    var user : User? = null
)
data class Address(
    @PrimaryKey(autoGenerate = true) var id: Int = 0,
    var pincode: String? = null,
    val ids: List<Int>,
    var isdefault : Int = 0
)
'''

SERVICE_KT = '''package com.x.utils
class WebServiceUtils {
    companion object {
        fun createUser(dm: DataManager, user: User) {
            // val url = Constants.BASE_URL + "old/removedEndpoint"
            /* val url = Constants.BASE_URL + "old/blockRemoved" */
            val url = Constants.BASE_URL + AppConstants.URL_CREATE_USER
            dm.retrofitInterface.createUser(url, user).enqueue(cb)
        }

        fun listOrders(dm: DataManager, token: String) {
            val data = HashMap<String, String>()
            data["storeId"] = "1"
            val url = Constants.BASE_URL + AppConstants.URL_LIST
            dm.retrofitInterface.list(url, token, data)
        }

        fun search(dm: DataManager, type: String) {
            val cc = CommonCategory()
            var url = ""
            when (type) {
                "b" -> { url = Constants.BASE_URL + "category/searchBrandX" }
                "c" -> { url = Constants.BASE_URL + "category/searchCat" }
            }
            dm.retrofitInterface.search(url, cc)
        }

        fun addOrUpdate(a: Address, dm: DataManager) {
            val url = Constants.BASE_URL + "user/" + if (a.id != 0) {
                "updateAddress"
            } else "addAddress"
            dm.retrofitInterface.addUpdate(url, a)
        }

        fun sendOtp(dm: DataManager, endUrl: String, data: HashMap<String, String>) {
            data.put("deviceType", "ANDROID")
            val url = Constants.BASE_URL + endUrl
            dm.retrofitInterface.sendOtp(url, data)
        }

        fun askOtp(dm: DataManager, m: HashMap<String, String>) {
            sendOtp(dm, AppConstants.URL_SEND_OTP, m)
            sendOtp(dm, AppConstants.URL_VERIFY, m)
        }

        fun generic(dm: DataManager, token: String) {
            val url = Constants.BASE_URL + AppConstants.URL_GENERIC
            val req = ProductListRequest(offset = 0, limit = 5, query_type = "deliverydashboard", response_key = "dash")
            dm.retrofitInterface.getGeneric(url, req, token)
        }

        fun genericGet(dm: DataManager, token: String) {
            val url = Constants.BASE_URL + AppConstants.URL_GENERIC
            dm.retrofitInterface.getGeneric(url, "Bearer " + token)
        }

        fun upload(dm: DataManager, file: File, id: RequestBody) {
            val url = Constants.BASE_URL + AppConstants.URL_UPLOAD
            val body = MultipartBody.Part.createFormData("sampleFile", file.name, rf)
            dm.retrofitInterface.upload(url, id, body)
        }

        fun pincode(dm: DataManager, info: AddressInfo) {
            val url = "https://api.postalpincode.in/pincode/" + "" + info.pincode
            dm.retrofitInterface.ext(url)
        }

        fun collision(dm: DataManager) {
            dm.orderHelper.getGeneric(dm)   // same name, different class and arity: not a Retrofit call
        }
    }
}
'''


@pytest.fixture
def android(tmp_path):
    root = tmp_path / "shop"
    write(root, "app/src/main/java/com/x/network/RetrofitInterface.kt", API_KT)
    write(root, "app/src/main/java/com/x/utils/AppConstants.kt", CONSTANTS_KT)
    write(root, "app/src/main/java/com/x/model/Models.kt", MODELS_KT)
    write(root, "app/src/main/java/com/x/utils/WebServiceUtils.kt", SERVICE_KT)
    write(root, "app/build.gradle.kts",
          'productFlavors { create("prod") { buildConfigField("String", "BASE_URL", "\\"https://api.shop.test/api/\\"") } }')
    write(root, "keystore.properties", "storePassword=hunter2hunter2\n")
    return root


@pytest.fixture
def scan(android):
    result = scan_project(FileAccessPolicy([android]), android)
    return result, {(e.method, e.path, e.variant): e for e in result.endpoints}


# ------------------------------------------------------------ small parsers
def test_split_top_level_respects_generics_and_strings():
    assert split_top_level('a: Map<String, String>, b: Int = foo(1, 2), c: String = "x,y"') == [
        "a: Map<String, String>", "b: Int = foo(1, 2)", 'c: String = "x,y"']


def test_strip_comments_keeps_urls_inside_strings():
    out = strip_comments('val a = "https://x.com/a" // trailing\n/* block\nval b */ val c = 1')
    assert "https://x.com/a" in out and "trailing" not in out and "val b" not in out and "val c = 1" in out


def test_parse_retrofit_interface_methods():
    methods = {m.name: m for m in parse_retrofit_interface(API_KT, "Api.kt") if m.name != "getGeneric"}
    assert methods["searchBrand"].path == "category/searchBrand" and methods["searchBrand"].body_type == "CommonCategory"
    assert methods["createUser"].path is None and methods["createUser"].http == "POST"
    assert methods["list"].has_auth and methods["list"].body_type == "Map<String,String>"
    assert methods["upload"].multipart and methods["ext"].http == "GET"
    assert methods["item"].path == "items/{id}"


# ---------------------------------------------------------------- Retrofit
def test_base_url_comes_from_gradle_flavor(scan):
    assert scan[0].base_url == "https://api.shop.test/api"


def test_at_url_method_is_paired_with_constant_and_body_model(scan):
    ep = scan[1][("POST", "user/createUser", "")]
    assert ep.auth is False and ep.body_type == "User"
    assert ep.body["user_id"] == 0 and ep.body["name"] == "guest"
    assert ep.body["allow_radius_in_meter"] == 2000  # @SerializedName alias
    assert ep.body["role"] is None  # an explicit null default is kept
    assert ep.body["kind"] == "Admin"  # no literal default: enum -> its first value
    assert ep.body["tags"] is None  # `= null` default is kept
    assert "cache" not in ep.body and "computed" not in ep.body and "lazyOne" not in ep.body  # transient/computed


def test_commented_out_urls_are_ignored(scan):
    paths = {e.path for e in scan[0].endpoints}
    assert "old/removedEndpoint" not in paths and "old/blockRemoved" not in paths


def test_literal_path_method_gets_its_body(scan):
    ep = scan[1][("POST", "category/searchBrand", "")]
    assert ep.body == {"id": 0, "title": None, "search": ""}


def test_map_body_keys_are_taken_from_the_call_site(scan):
    ep = scan[1][("POST", "cart/list", "")]
    assert ep.auth is True and ep.body == {"storeId": "string"}


def test_url_reassigned_in_when_branches_gives_every_branch(scan):
    paths = {e.path for e in scan[0].endpoints}
    assert {"category/searchBrandX", "category/searchCat"} <= paths


def test_computed_url_with_if_else_expands_both_paths(scan):
    paths = {e.path for e in scan[0].endpoints}
    assert {"user/updateAddress", "user/addAddress"} <= paths
    addr = scan[1][("POST", "user/addAddress", "")].body
    assert addr["pincode"] is None and addr["ids"] == [0]  # no default -> sample from the List<Int> type


def test_wrapper_function_url_parameter_is_followed_to_callers(scan):
    for path in ("user/sendOtp", "user/verifyOtp"):
        ep = scan[1][("POST", path, "")]
        assert ep.body == {"deviceType": "string"}


def test_overloads_are_chosen_by_argument_count_and_token_position(scan):
    ep = scan[1][("POST", "data/generic", "deliverydashboard")]
    assert ep.body_type == "ProductListRequest" and ep.body["query_type"] == "deliverydashboard"
    assert ep.body["response_key"] == "dash" and ep.body["screentype"] == "home"
    # the GET overload of the same helper must not create a second, bogus GET endpoint
    assert not any(e.method == "GET" and e.path == "data/generic" for e in scan[0].endpoints)
    assert any("also calls this URL with GET" in n for n in ep.notes)


def test_multipart_form_uses_the_field_names_from_the_code(scan):
    ep = scan[1][("POST", "uploader/uploadImage", "")]
    assert ep.body is None
    assert ep.form == [("imageid", "text", "example"), ("sampleFile", "file", "")]


def test_external_api_is_kept_as_an_absolute_url(scan):
    ep = scan[1][("GET", "https://api.postalpincode.in/pincode/{{pincode}}", "")]
    assert ep.folder == "External" and any("third-party" in n for n in ep.notes)


def test_path_and_query_params(scan):
    ep = scan[1][("GET", "items/{id}", "")]
    assert ep.query == {"expand": ""}


def test_unused_methods_are_reported_but_name_collisions_are_not_calls(scan):
    warnings = " ".join(scan[0].warnings)
    assert "never called" in warnings and "neverUsed" in warnings
    assert "getGeneric" not in warnings.split("never called")[-1]  # the orderHelper.getGeneric(dm) collision
    assert "could not be worked out" not in warnings


def test_url_constant_without_call_site_is_added_as_a_guess(scan):
    ep = scan[1][("POST", "misc/neverCalled", "")]
    assert ep.body is None and any("guess" in n for n in ep.notes)
    assert not any(e.path == "image/jpeg" for e in scan[0].endpoints)  # mime types are not routes


def test_large_models_are_capped(tmp_path):
    root = tmp_path / "big"
    fields = ",\n".join(f"    var f{i}: Int = 0" for i in range(80))
    write(root, "Api.kt", 'import retrofit2.http.*\ninterface A { @POST("big/save") fun save(@Body b: Big): Call<R> }\n')
    write(root, "Big.kt", f"data class Big(\n{fields}\n)\n")
    ep = {e.path: e for e in scan_project(FileAccessPolicy([root]), root).endpoints}["big/save"]
    assert len(ep.body) == 45 and any("more than 45 fields" in n for n in ep.notes)


def test_scanner_never_reads_secret_files(android):
    write(android, "app/src/main/java/com/x/utils/Keys.kt", 'val URL_SECRET_API = "should/notappear"\n')
    (android / "app/src/main/java/com/x/utils/Keys.kt").rename(android / "app/src/main/java/com/x/utils/secrets.properties")
    paths = {e.path for e in scan_project(FileAccessPolicy([android]), android).endpoints}
    assert "should/notappear" not in paths


# ------------------------------------------------------------------ models
def test_model_index_nested_and_recursive_types():
    idx = ModelIndex()
    idx.add_source("data class Node(val name: String, val child: Node?, val kids: List<Node>, val owner: Owner)\n"
                   "data class Owner(val id: Long, val tags: Map<String, String>)\n", ".kt")
    sample = idx.sample_class("Node")[0]
    assert sample["name"] == "string" and sample["owner"] == {"id": 0, "tags": {}}
    assert isinstance(sample["child"], dict) and isinstance(sample["kids"], list)  # recursion is depth-limited, not infinite


def test_java_models():
    idx = ModelIndex()
    idx.add_source('public class Item {\n  private String name;\n  private int qty = 1;\n  private static int COUNT = 0;\n'
                   '  @SerializedName("unit_price")\n  private double price;\n}\n', ".java")
    assert idx.sample_class("Item")[0] == {"name": "string", "qty": 1, "unit_price": 0.0}


# ----------------------------------------------------------- other frameworks
def test_express_routes_with_mount_prefix_and_body_keys(tmp_path):
    root = tmp_path / "web"
    write(root, "server.js", "const userRoutes = require('./routes/user.routes');\napp.use('/api/user', userRoutes);\napp.listen(4000);\n")
    write(root, "routes/user.routes.js",
          "router.post('/login', async (req, res) => { const { email, password } = req.body; });\nrouter.get('/list', (req, res) => {});\n")
    result = scan_project(FileAccessPolicy([root]), root)
    eps = {(e.method, e.path): e for e in result.endpoints}
    assert eps[("POST", "api/user/login")].body == {"email": "string", "password": "string"}
    assert ("GET", "api/user/list") in eps and result.base_url == "http://localhost:4000"


def test_fastapi_routes_with_pydantic_body(tmp_path):
    root = tmp_path / "svc"
    write(root, "app/api/chat.py", '''from fastapi import APIRouter
from pydantic import BaseModel
router = APIRouter(prefix="/api", tags=["chat"])

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str
    limit: int = 5
    tags: list[str] = []

@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    pass

@router.get("/health")
async def health(request: Request):
    pass
''')
    write(root, "app/main.py", "from fastapi import FastAPI\napp = FastAPI()\n")
    result = scan_project(FileAccessPolicy([root]), root)
    eps = {(e.method, e.path): e for e in result.endpoints}
    assert eps[("POST", "api/chat")].body == {"conversation_id": None, "message": "string", "limit": 5, "tags": []}
    assert eps[("GET", "api/health")].body is None and result.base_url == "http://localhost:8000"


def test_flask_routes(tmp_path):
    root = tmp_path / "flask"
    write(root, "app.py", "from flask import Flask\napp = Flask(__name__)\n@app.route('/ping')\ndef ping():\n    pass\n"
                          "@app.route('/save', methods=['POST', 'PUT'])\ndef save():\n    pass\n")
    keys = {(e.method, e.path) for e in scan_project(FileAccessPolicy([root]), root).endpoints}
    assert {("GET", "ping"), ("POST", "save"), ("PUT", "save")} <= keys


def test_spring_controller(tmp_path):
    root = tmp_path / "spring"
    write(root, "ItemController.java", '''@RestController
@RequestMapping("/api/items")
public class ItemController {
    @PostMapping("/add")
    public Item add(@RequestBody ItemDto dto) { return null; }

    @GetMapping("/{id}")
    public Item get(@PathVariable Long id) { return null; }
}
''')
    write(root, "ItemDto.java", "public class ItemDto {\n  private String name;\n  private int qty = 2;\n}\n")
    eps = {(e.method, e.path): e for e in scan_project(FileAccessPolicy([root]), root).endpoints}
    assert eps[("POST", "api/items/add")].body == {"name": "string", "qty": 2}
    assert ("GET", "api/items/{id}") in eps


def test_openapi_json(tmp_path):
    root = tmp_path / "spec"
    write(root, "openapi.json", json.dumps({
        "openapi": "3.0.0",
        "paths": {"/pets": {"post": {"operationId": "addPet", "summary": "Add a pet", "security": [{"b": []}],
                                     "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}}}},
                            "get": {"parameters": [{"name": "limit", "in": "query"}]}}},
        "components": {"schemas": {"Pet": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}}},
    }))
    eps = {(e.method, e.path): e for e in scan_project(FileAccessPolicy([root]), root).endpoints}
    assert eps[("POST", "pets")].body == {"name": "string", "age": 0} and eps[("POST", "pets")].auth is True
    assert eps[("GET", "pets")].query == {"limit": ""}


DOCS = '''# API
**Base URL:** `https://api.docs.test/api/`

## 1. User (`/api/user`)
| Method | Endpoint | Auth | Status | Description |
|--------|----------|------|--------|-------------|
| POST | `user/login` | No | ✅ Active | Log in |
| POST | `user/delete` | Yes | 🔧 Backend | Delete |
| GET | `user/list` | No | ⚠️ Deprecated | List |

## 2. Dashboard
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `dashboard/home` | No | Home data |

### Not mounted
| Endpoint | Description |
|----------|-------------|
| inventory/temp | unreachable |

### Known `query_type` values (`data/genericApiData`)
| `query_type` | `response_key` | Status | Notes |
|---|---|---|---|
| `deliverydashboard` | `deliveryDashboard` | ✅ Active | x |
| `store_search` | `storeList` (placeholder) | 🔧 Backend (Pending) | y |
'''


def test_markdown_docs_tables(tmp_path):
    root = tmp_path / "docs"
    write(root, "API_DOCUMENTATION.md", DOCS)
    result = scan_project(FileAccessPolicy([root]), root)
    eps = {(e.method, e.path, e.variant): e for e in result.endpoints}
    assert result.base_url == "https://api.docs.test/api"
    login = eps[("POST", "user/login", "")]
    assert login.auth is False and login.status == "active" and login.description == "Log in"
    assert eps[("POST", "user/delete", "")].auth is True and eps[("POST", "user/delete", "")].status == "backend/admin"
    assert eps[("GET", "user/list", "")].status == "deprecated"
    assert ("POST", "dashboard/home", "") in eps  # table without a Status column
    assert not any("inventory" in e.path for e in result.endpoints)  # unmounted routers are not requests
    assert eps[("POST", "data/genericApiData", "deliverydashboard")].body == {"query_type": "deliverydashboard", "response_key": "deliveryDashboard"}
    assert not any(e.variant == "query_type" for e in result.endpoints)  # the table's header row is not a value
    assert all(not e.from_code for e in result.endpoints) and any("only from the docs" in w for w in result.warnings)


def test_docs_and_code_are_merged_docs_win_on_auth(tmp_path):
    root = tmp_path / "both"
    write(root, "API.md", DOCS)
    write(root, "Api.kt", 'import retrofit2.http.*\ninterface A { @POST("user/login") fun login(@Body u: U): Call<R> }\ndata class U(val email: String = "a@b.c")\n')
    login = {(e.method, e.path): e for e in scan_project(FileAccessPolicy([root]), root).endpoints}[("POST", "user/login")]
    assert login.body == {"email": "a@b.c"} and login.description == "Log in" and login.status == "active"
    assert len(login.sources) == 2


def test_empty_project(tmp_path):
    root = tmp_path / "empty"
    write(root, "README.md", "hi")
    result = scan_project(FileAccessPolicy([root]), root)
    assert result.endpoints == [] and any("No API endpoints" in w for w in result.warnings)


# ------------------------------------------------------------------ Postman
@pytest.fixture
def collection(scan):
    result, _ = scan
    return build_collection("Shop API", result.endpoints, result.base_url)


def find_item(col, path_suffix, method="POST"):
    def rec(items):
        for it in items:
            if "item" in it:
                found = rec(it["item"])
                if found:
                    return found
            elif it["request"]["url"]["raw"].endswith(path_suffix) and it["request"]["method"] == method:
                return it
    return rec(col["item"])


def test_collection_is_valid_v21_with_variables_and_folders(collection):
    assert collection["info"]["schema"].endswith("v2.1.0/collection.json")
    assert {v["key"] for v in collection["variable"]} == {"baseUrl", "token"}
    assert collection["auth"]["bearer"][0]["value"] == "{{token}}"
    assert {f["name"] for f in collection["item"]} >= {"User", "Cart", "Category", "Data", "Uploader", "External"}
    json.dumps(collection)


def test_every_request_is_well_formed(collection):
    def walk(items):
        for it in items:
            if "item" in it:
                yield from walk(it["item"])
            else:
                yield it
    count = 0
    for it in walk(collection["item"]):
        count += 1
        req = it["request"]
        assert req["url"]["raw"].startswith(("{{baseUrl}}/", "http")) and req["url"]["host"]
        if req.get("body", {}).get("mode") == "raw":
            json.loads(req["body"]["raw"])
    assert count == 14  # exactly the endpoints in the sample project: no bogus extras


def test_login_request_saves_the_token_and_is_unauthenticated(collection):
    item = find_item(collection, "user/createUser")
    assert item["request"]["auth"] == {"type": "noauth"}
    script = "\n".join(item["event"][0]["script"]["exec"])
    assert "pm.collectionVariables.set('token'" in script
    assert "event" not in find_item(collection, "cart/list")


def test_authenticated_requests_inherit_bearer_auth(collection):
    assert "auth" not in find_item(collection, "cart/list")["request"]


def test_form_data_body(collection):
    body = find_item(collection, "uploader/uploadImage")["request"]["body"]
    assert body["mode"] == "formdata"
    assert {"key": "sampleFile", "type": "file", "src": []} in body["formdata"]


def test_path_params_query_and_external_urls(collection):
    item = find_item(collection, "items/:id?expand=", "GET")
    assert item["request"]["url"]["path"] == ["items", ":id"]
    assert item["request"]["url"]["query"] == [{"key": "expand", "value": ""}]
    ext = find_item(collection, "{{pincode}}", "GET")
    assert ext["request"]["url"]["host"] == ["api", "postalpincode", "in"] and ext["request"]["url"]["protocol"] == "https"


def test_variant_requests_are_named_and_documented(collection):
    item = find_item(collection, "data/generic")
    assert item["name"] == "generic [deliverydashboard]"
    assert "Body type: ProductListRequest" in item["request"]["description"]


def test_map_bodies_are_flagged_as_partial(collection):
    assert "key/value map" in find_item(collection, "user/sendOtp")["request"]["description"]


def test_duplicate_names_in_a_folder_are_disambiguated(collection):
    names = [it["name"] for f in collection["item"] if f["name"] == "User" for it in f["item"]]
    assert len(names) == len(set(names))


def test_display_name_and_slug():
    from app.tools.api_scanner import Endpoint
    assert display_name(Endpoint("GET", "items/{id}")) == "items"
    assert display_name(Endpoint("GET", "https://x.io/pincode/{{pincode}}")) == "pincode"
    assert slugify("Rasanpani API!") == "rasanpani-api" and slugify("///") == "api"


# --------------------------------------------------------------------- tool
async def test_tool_writes_collection_and_returns_download_link(android, tmp_path):
    exports = tmp_path / "exports"
    tool = GeneratePostmanCollectionTool(FileAccessPolicy([android]), exports, "http://127.0.0.1:8000")
    result = await tool.execute(project_path=str(android))
    assert result.ok and "requests in" in result.output and "Base URL: https://api.shop.test/api" in result.output
    assert result.files == [{"title": "shop-api.postman_collection.json", "url": "http://127.0.0.1:8000/api/exports/shop-api.postman_collection.json"}]
    data = json.loads((exports / "shop-api.postman_collection.json").read_text())
    assert data["info"]["name"] == "shop API"
    assert "hunter2" not in json.dumps(data)


async def test_tool_custom_name_and_refusals(android, tmp_path):
    tool = GeneratePostmanCollectionTool(FileAccessPolicy([android]), tmp_path / "exports", "http://127.0.0.1:8000")
    assert (await tool.execute(project_path=str(android), name="My Shop")).files[0]["title"] == "my-shop.postman_collection.json"
    for bad in ("/etc", str(android / "keystore.properties"), str(android / "app/build.gradle.kts"), ""):
        assert not (await tool.execute(project_path=bad)).ok


async def test_tool_reports_projects_without_an_api(tmp_path):
    root = tmp_path / "plain"
    write(root, "README.md", "nothing here")
    tool = GeneratePostmanCollectionTool(FileAccessPolicy([root]), tmp_path / "exports", "http://127.0.0.1:8000")
    result = await tool.execute(project_path=str(root))
    assert not result.ok and "No API endpoints" in result.error
    assert not (tmp_path / "exports").exists()


# ------------------------------------------------------------ naming / folders
def test_folders_skip_generic_api_prefixes():
    from app.tools.api_scanner import Endpoint
    assert Endpoint("GET", "api/products").folder == "Products"
    assert Endpoint("GET", "api/v1/orders/list").folder == "Orders"
    assert Endpoint("GET", "user/login").folder == "User"
    assert Endpoint("GET", "health").folder == "General"
    assert Endpoint("GET", "api/{id}").folder == "General"
    assert Endpoint("GET", "https://x.io/a/b").folder == "External"


def test_request_names_by_style():
    from app.tools.api_scanner import Endpoint
    fast = Endpoint("POST", "api/products", name="create_product", sources=["fastapi app/routes/products.py#create_product"])
    spring = Endpoint("GET", "api/items/{id}", name="getItemById", sources=["spring ItemController.java#getItemById"])
    express = Endpoint("POST", "api/user/login", name="login", sources=["express routes/user.routes.js"])
    rpc = Endpoint("POST", "user/createUser", name="loginRegister", sources=["retrofit Api.kt#loginRegister"])
    assert display_name(fast) == "Create product" and display_name(spring) == "Get item by id"
    assert display_name(express) == "POST /api/user/login"
    assert display_name(rpc) == "createUser"


def test_fastapi_project_collection_is_readable(tmp_path):
    root = tmp_path / "svc"
    write(root, "app/routes/products.py", '''from fastapi import APIRouter
from pydantic import BaseModel
router = APIRouter(prefix="/api/products")

class ProductIn(BaseModel):
    name: str
    price: float

@router.get("")
async def list_products():
    pass

@router.post("")
async def create_product(body: ProductIn):
    pass

@router.get("/{product_id}")
async def get_product(product_id: int):
    pass
''')
    result = scan_project(FileAccessPolicy([root]), root)
    col = build_collection("Svc", result.endpoints, result.base_url)
    assert [f["name"] for f in col["item"]] == ["Products"]
    assert [i["name"] for i in col["item"][0]["item"]] == ["List products", "Create product", "Get product"]


def test_summary_is_honest_about_login(tmp_path):
    from app.tools.postman import summarise
    root = tmp_path / "svc"
    write(root, "app.py", "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/ping')\nasync def ping():\n    pass\n")
    result = scan_project(FileAccessPolicy([root]), root)
    text = summarise(result, build_collection("Svc", result.endpoints, result.base_url))
    assert "no login endpoint was found" in text and "set automatically" not in text
    write(root, "auth.py", "from fastapi import APIRouter\nrouter = APIRouter()\n@router.post('/login')\nasync def login():\n    pass\n")
    result = scan_project(FileAccessPolicy([root]), root)
    assert "saved automatically by the login requests" in summarise(result, build_collection("Svc", result.endpoints, result.base_url))
