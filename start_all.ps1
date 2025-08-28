# Start All Foresight Services

# Open VS Code in project root
Start-Process code "C:\Users\Asus\foresight"

# Backend (FastAPI) tab
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\Users\Asus\foresight; .venv\Scripts\activate; uvicorn main:app --reload"

# Frontend (React) tab
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\Users\Asus\foresight\src\ui\dashboard; npm run dev"

# scrcpy (phone mirror) tab
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\Users\Asus\foresight; scrcpy --window-title 'DJI_MIRROR' --max-fps=30 --stay-awake --turn-screen-off"
