# VHS tape file for recording the diffusion vs GPT race
# https://github.com/charmbracelet/vhs

Output assets/demo.gif
Output assets/demo.webm

Set Shell zsh
Set FontSize 14
Set Width 1000
Set Height 650
Set Padding 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 0

# Hide the command execution and model loading
Hide
Type "python visualize.py --delay 0.02"
Enter
Sleep 9s
Enter
Show

# Race: 225 frames × 0.02s ≈ 4.5s + buffer
Sleep 6s

# Hold on comparison summary for viewers to read
Sleep 6s
