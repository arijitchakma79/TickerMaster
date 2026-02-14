.PHONY: backend-install backend-run frontend-install frontend-run

backend-install:
	cd backend && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

backend-run:
	cd backend && . .venv/bin/activate && uvicorn app.main:app --reload --port 8000

frontend-install:
	cd frontend && npm install

frontend-run:
	cd frontend && npm run dev
