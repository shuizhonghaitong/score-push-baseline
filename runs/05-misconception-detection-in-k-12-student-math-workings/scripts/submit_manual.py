"""Submit a competition file using the kagglesdk flow, with the upload host swapped
to storage.googleapis.com (www.googleapis.com is proxy-blocked).
Usage: python3 submit_manual.py <file> [message]
"""
import sys, os
sys.path.insert(0, "/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle")
from kaggle.api.kaggle_api_extended import KaggleApi, ResumableUploadResult
from kagglesdk.competitions.types.competition_api_service import (
    ApiStartSubmissionUploadRequest, ApiCreateSubmissionRequest)

COMP = "misconception-detection-in-k-12-student-math-workings"

def main():
    path = sys.argv[1]
    msg = sys.argv[2] if len(sys.argv) > 2 else ""
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as kaggle:
        req = ApiStartSubmissionUploadRequest()
        req.competition_name = COMP
        req.file_name = os.path.basename(path)
        req.content_length = os.path.getsize(path)
        req.last_modified_epoch_seconds = int(os.path.getmtime(path))
        response = kaggle.competitions.competition_api_client.start_submission_upload(req)
        url = response.create_url.replace("www.googleapis.com", "storage.googleapis.com")
        status = api.upload_complete(path, url, quiet=True)
        if status != ResumableUploadResult.COMPLETE:
            print("upload failed:", status)
            return 1
        sreq = ApiCreateSubmissionRequest()
        sreq.competition_name = COMP
        sreq.blob_file_tokens = response.token
        if msg:
            sreq.submission_description = msg
        sresp = kaggle.competitions.competition_api_client.create_submission(sreq)
    print("submission:", sresp)
    return 0

if __name__ == "__main__":
    sys.exit(main())
