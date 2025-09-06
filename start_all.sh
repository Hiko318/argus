#!/bin/bash
# Start All Foresight Services for Unix/Linux/Jetson
# Usage: ./start_all.sh
# Make executable: chmod +x start_all.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
UI_PATH="$PROJECT_ROOT/src/ui/dashboard"
LOG_DIR="$PROJECT_ROOT/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Starting Foresight SAR System...${NC}"
echo -e "${BLUE}Project Root: $PROJECT_ROOT${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -i :"$1" >/dev/null 2>&1
}

# Function to start service in background with logging
start_service() {
    local name="$1"
    local command="$2"
    local log_file="$LOG_DIR/${name}.log"
    
    echo -e "${YELLOW}Starting $name...${NC}"
    nohup bash -c "$command" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$LOG_DIR/${name}.pid"
    echo -e "${GREEN}$name started (PID: $pid)${NC}"
    echo -e "${BLUE}Log: $log_file${NC}"
}

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    
    # Kill all background processes
    for pid_file in "$LOG_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            local name=$(basename "$pid_file" .pid)
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${YELLOW}Stopping $name (PID: $pid)...${NC}"
                kill "$pid" 2>/dev/null || true
                sleep 1
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
    
    echo -e "${GREEN}All services stopped.${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Python virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip
    
    # Install appropriate requirements based on platform
    if command_exists jetson_release; then
        echo -e "${YELLOW}Detected Jetson device, installing Jetson-optimized requirements...${NC}"
        pip install -r requirements-jetson.txt
    else
        echo -e "${YELLOW}Installing standard requirements...${NC}"
        pip install -r requirements.txt
    fi
else
    echo -e "${GREEN}Virtual environment found${NC}"
fi

# Check Node.js and npm
if ! command_exists node; then
    echo -e "${RED}Node.js not found. Please install Node.js 18+ and npm${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}npm not found. Please install npm${NC}"
    exit 1
fi

# Check if UI dependencies are installed
if [ ! -d "$UI_PATH/node_modules" ]; then
    echo -e "${YELLOW}Installing UI dependencies...${NC}"
    cd "$UI_PATH"
    npm install
    cd "$PROJECT_ROOT"
fi

# Check for port conflicts
if port_in_use 8000; then
    echo -e "${RED}Port 8000 is already in use. Please stop the conflicting service.${NC}"
    exit 1
fi

if port_in_use 5173; then
    echo -e "${RED}Port 5173 is already in use. Please stop the conflicting service.${NC}"
    exit 1
fi

echo -e "${GREEN}Prerequisites check passed${NC}"
echo ""

# Start Backend (FastAPI)
backend_cmd="cd '$PROJECT_ROOT' && source '$VENV_PATH/bin/activate' && python main.py --web-server"
start_service "backend" "$backend_cmd"

# Wait a moment for backend to start
sleep 3

# Start Frontend (React)
frontend_cmd="cd '$UI_PATH' && npm run dev -- --host 0.0.0.0"
start_service "frontend" "$frontend_cmd"

# Start scrcpy if available (for phone mirroring)
if command_exists scrcpy; then
    echo -e "${YELLOW}Starting scrcpy for phone mirroring...${NC}"
    scrcpy_cmd="scrcpy --window-title='DJI_MIRROR' --max-fps=30 --stay-awake --turn-screen-off"
    start_service "scrcpy" "$scrcpy_cmd"
else
    echo -e "${YELLOW}scrcpy not found. Phone mirroring will not be available.${NC}"
    echo -e "${BLUE}To install scrcpy: sudo apt install scrcpy${NC}"
fi

echo ""
echo -e "${GREEN}All services started successfully!${NC}"
echo -e "${BLUE}Backend API: http://localhost:8000${NC}"
echo -e "${BLUE}Frontend UI: http://localhost:5173${NC}"
echo -e "${BLUE}API Documentation: http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo -e "${BLUE}Logs are available in: $LOG_DIR${NC}"
echo ""

# Keep script running and monitor services
while true; do
    sleep 5
    
    # Check if any service has died
    for pid_file in "$LOG_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            local name=$(basename "$pid_file" .pid)
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "${RED}Service $name (PID: $pid) has stopped unexpectedly${NC}"
                echo -e "${BLUE}Check log: $LOG_DIR/${name}.log${NC}"
                rm -f "$pid_file"
            fi
        fi
    done
done