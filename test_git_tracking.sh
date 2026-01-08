#!/bin/bash
# Test script for git tracking functionality

echo "=== Testing Git Code Modification Tracking ==="
echo ""

# Check if we're in a git repo
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "✓ In a git repository"
else
    echo "✗ Not in a git repository"
    echo "  Run: git init"
    exit 1
fi

# Test the git_code_tracker module
echo ""
echo "Testing git_code_tracker.py..."
python3 git_code_tracker.py /workspace/MPM_pytorch

echo ""
echo "Current git status:"
git status --short

echo ""
echo "Recent commits (last 5):"
git log --oneline -5

echo ""
echo "Commits from Claude Code Modification System:"
git log --grep="Claude Code Modification" --oneline -5

echo ""
echo "=== Test Complete ==="
echo ""
echo "To test with a real modification:"
echo "1. Edit src/MPM_pytorch/models/Siren_Network.py"
echo "2. Add a comment or small change"
echo "3. Run: python git_code_tracker.py /workspace/MPM_pytorch"
echo "4. Check: git log -1 --stat"
