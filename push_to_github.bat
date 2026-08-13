@echo off
git init
git branch -M main
git remote add origin https://github.com/danielalejandromendezorozco4-crypto/An-fyttest.git 2>nul
git add .
git commit -m "Refactorizacion modular completa de An-FyT (config, data, engine, services, reports, tests, app.py)"
git push -u origin main --force
