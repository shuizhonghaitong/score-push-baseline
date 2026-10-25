"""Test: can a Kaggle kernel authenticate to the Kaggle API and submit?"""
import os, json, warnings
warnings.filterwarnings('ignore')
print('env check: HOME=%s' % os.environ.get('HOME'))
print('kaggle.json exists:', os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')))
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print('AUTH_OK')
    # whoami-ish: list my active kernels (read op)
    try:
        with api.build_kaggle_client() as kaggle:
            from kagglesdk.kernels.types.kernel_api_service import ListKernelsRequest
            resp = kaggle.kernels.kernel_api_client.list_kernels(ListKernelsRequest())
        print('LIST_OK count=', len(resp.results))
    except Exception as e:
        print('LIST_ERR', type(e).__name__, str(e)[:200])
except Exception as e:
    print('AUTH_FAIL', type(e).__name__, str(e)[:400])
