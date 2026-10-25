"""Small helper around the kagglesdk client for kernel inspection."""
import sys

def client():
    from kagglehub.datasets_helpers import build_kaggle_client
    return build_kaggle_client()

def _req(cls, user, slug):
    r = cls()
    r.user_name = user
    r.kernel_slug = slug
    return r

def get_kernel(user, slug):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    r = client().kernels.kernels_api_client.get_kernel(_req(ApiGetKernelRequest, user, slug))
    m = r.metadata
    return {
        "ref": f"{user}/{slug}",
        "title": m.title,
        "score": getattr(m, "score", None),
        "last_run_time": str(getattr(m, "last_run_time", "")),
        "is_private": getattr(m, "is_private", None),
    }

def session_status(user, slug):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
    return client().kernels.kernels_api_client.get_kernel_session_status(
        _req(ApiGetKernelSessionStatusRequest, user, slug)
    )

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "get":
        import json
        user, slug = sys.argv[2].split("/", 1)
        print(json.dumps(get_kernel(user, slug), indent=1, default=str))
    elif mode == "status":
        user, slug = sys.argv[2].split("/", 1)
        print(session_status(user, slug).status)
