#!/bin/bash
# C3 全量打分 tmux 启动脚本
# 用法: bash start.sh [--n 600] [--workers 32] [--model deepseek-v4-pro]

SESSION="c3-scoring"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' 已存在，直接 attach"
    tmux attach -t "$SESSION"
    exit 0
fi

# 创建宽屏 session（适配仪表盘显示）
tmux new-session -d -s "$SESSION" -x 220 -y 55

# 主窗口：运行打分脚本
tmux rename-window -t "$SESSION:0" "scoring"
tmux send-keys -t "$SESSION:0" "python run_c3_full.py $*" Enter

# 第二窗口：实时 tail 日志
tmux new-window -t "$SESSION" -n "log"
tmux send-keys -t "$SESSION:log" "tail -f data/run.log" Enter

# 第三窗口：备用 shell
tmux new-window -t "$SESSION" -n "shell"

# 回到主窗口
tmux select-window -t "$SESSION:scoring"
tmux attach -t "$SESSION"
