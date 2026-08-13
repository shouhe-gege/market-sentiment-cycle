#!/bin/bash
# upload_v2_to_github.sh
# 一键把 V2 项目推到 GitHub
# 使用前请修改下面 2 个变量

USERNAME="shouhe-gege"
REPO_NAME="market-sentiment-cycle-v2"

set -e

echo "==> 初始化仓库"
git init
git config user.name "$USERNAME"
git config user.email "$USERNAME@users.noreply.github.com"

echo "==> 添加文件"
git add market_sentiment_cycle_v2.py _demo_data.py README_v2.md upload_v2_to_github.sh
git commit -m "feat: 市场情绪周期判断器 V2 - 实盘主线/龙头/中军识别"

echo "==> 关联远程仓库（请先在 GitHub 网页创建同名空仓库）"
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git"

echo "==> 推送到 GitHub"
echo "    首次推送可能需要输入 GitHub 用户名 + Personal Access Token"
git push -u origin main

echo ""
echo "✅ 完成！仓库地址: https://github.com/$USERNAME/$REPO_NAME"
