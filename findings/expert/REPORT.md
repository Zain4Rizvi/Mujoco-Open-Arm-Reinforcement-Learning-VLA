# Expert demonstrations

The training data is a scripted throw, not a learned policy. The script sees the true ball and bin positions. The network never does.

On 100 episodes the script put the ball in the target bin 87 times, missed the bin 9 times, put it in the wrong bin 3 times, and failed to grasp once. The saved summary records a mean throw error of 7.0 cm.

Every fine-tune below was trained only on successes from this script. Matching the expert's joint motion is a different problem from seeing which ball the sentence named.

## Videos

- `videos/success_green-into-purple.mp4` — a completed throw.
- `videos/missed_blue-into-green.mp4` — the script grasped the ball and missed the bin.

Source: `artifacts/expert/summary.json`.
