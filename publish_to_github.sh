#!/usr/bin/env bash
# Публикация репозитория на GitHub.
# Использование: bash publish_to_github.sh https://github.com/<ваш-логин>/<репозиторий>.git
# Предварительно создайте ПУСТОЙ репозиторий на github.com (кнопка New, без README/.gitignore).
set -euo pipefail
REMOTE="${1:?Укажите ссылку на пустой репозиторий GitHub}"
git init
git add .
git commit -m "Task 3: hierarchical cluster analysis (Ward, SPSS-style) of trade partners"
git branch -M main
git remote add origin "$REMOTE"
git push -u origin main
echo "Готово. Ссылка для сдачи: ${REMOTE%.git}"
