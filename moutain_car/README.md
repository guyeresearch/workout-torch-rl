Notes
```
for item in zip(critic_.parameters(),critic.parameters()):
          item[0].data = tau*item[1].data + (1-tau)*item[0].data
```
The above lines of code is the wrong way to copy weights from the actual network to the target network. Because tensor.data is a pass by reference. The weights in the target network will change with the actual network after the assignment.

run5.py works but run7.py does not. In Q-learning, the next step target value must be computed using the current policy not the exploring policy. run7.py tries to learn on-policy Q values using the exploring policy. As run5.py works, it suggets in Monte Carlo learning, off-policy returns can be used to compute on-policy Q values. Try to explore more why it is mathematically so.
