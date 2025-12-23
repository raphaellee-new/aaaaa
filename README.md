# GitHub Actions 실행 패키지 (Python Script)

이 패키지는 `Simple_Agent.ipynb`를 **순수 Python 스크립트**(`src/simple_agent.py`)로 변환해두었고,
GitHub Actions에서는 **Jupyter를 사용하지 않고** `python`으로만 실행합니다.

## 1) 레포지토리 배치 구조
아래 구조 그대로 레포지토리 루트에 커밋하세요.

```
<repo-root>/
  src/simple_agent.py
  requirements.txt
  .github/workflows/run_simple_agent.yml
```

## 2) GitHub Secrets 등록
Repository > Settings > Secrets and variables > Actions > New repository secret

사용 기능에 따라 필요한 Secret만 등록:
- MISTRAL_API_KEY
- SERVICE_KEY
- GMAIL_ADDRESS
- GMAIL_APP_PASSWORD
- TO_EMAIL

## 3) 실행
- 수동: Actions 탭 > Run Simple Agent (Python) > Run workflow
- 스케줄: `.github/workflows/run_simple_agent.yml`의 cron 수정 (cron은 UTC 기준)

## 4) 주의사항
변환된 `src/simple_agent.py`에 아래가 있으면 Actions에서 실패할 수 있습니다.
- `google.colab` 관련 import/사용
- `files.download()` 같은 Colab 전용 기능
- 노트북 매직/쉘 명령은 변환 시 제거되었지만, 남아있다면 수동 제거 필요
