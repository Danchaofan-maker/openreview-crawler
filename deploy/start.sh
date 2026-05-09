#!/bin/bash
# C3 全量打分 tmux 启动脚本
# 服务器规格: 2核 2GiB，--max-workers 64 是内存安全上限
# 用法: bash start.sh [额外参数透传给 run_c3_full.py]

SESSION="c3-scoring"

# 针对 2GiB 服务器的默认参数
DEFAULT_ARGS="--init-workers 8 --max-workers 64 --min-workers 4"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' 已存在，直接 attach"
    tmux attach -t "$SESSION"
    exit 0
fi

# 创建宽屏 session（适配仪表盘显示）
tmux new-session -d -s "$SESSION" -x 220 -y 55

# 主窗口：运行打分脚本
tmux rename-window -t "$SESSION:0" "scoring"
tmux send-keys -t "$SESSION:0" "python run_c3_full.py $DEFAULT_ARGS $*" Enter

# 第二窗口：实时 tail 日志
tmux new-window -t "$SESSION" -n "log"
tmux send-keys -t "$SESSION:log" "tail -f data/run.log" Enter

# 第三窗口：备用 shell
tmux new-window -t "$SESSION" -n "shell"

# 回到主窗口
tmux select-window -t "$SESSION:scoring"
tmux attach -t "$SESSION"
