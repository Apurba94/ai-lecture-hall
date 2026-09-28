@echo off
REM Publish changes: build check -> commit -> push to GitHub -> Vercel deploys automatically.
REM Usage: double-click, or run:  publish "Add lecture on graph transformers"
cd /d "%~dp0"

echo Checking that the website builds...
node build.js
if errorlevel 1 (
  echo.
  echo Build failed - nothing was published. Fix the error shown above and try again.
  pause
  exit /b 1
)

git add -A
git diff --cached --quiet
if not errorlevel 1 (
  echo.
  echo No changes to publish.
  pause
  exit /b 0
)

set "MSG=%~1"
if "%MSG%"=="" set "MSG=Update blog content"
git commit -m "%MSG%"
git push
if errorlevel 1 (
  echo.
  echo Push to GitHub failed. Check your internet connection or GitHub login.
  pause
  exit /b 1
)

echo.
echo Published to GitHub. Vercel will update https://ai-lecture-hall.vercel.app in about a minute.
pause
