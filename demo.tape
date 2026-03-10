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

# Hide everything: command, model loading, precomputation, "Press Enter"
Hide
Type "python visualize.py --delay 0.035"
Enter
Sleep 9s
Enter
Show

# Race: 225 frames × 0.035s ≈ 7.9s + buffer
Sleep 9s

# Hide the comparison summary — we only want the race
Hide
Sleep 3s
