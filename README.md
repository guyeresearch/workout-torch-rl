# My project's README
```
for item in zip(critic_.parameters(),critic.parameters()):
          item[0].data = tau*item[1].data + (1-tau)*item[0].data
```
The above lines of code is the wrong way to copy weights from the actual network to the target network. Because tensor.data is a pass by reference. The weights in the target network will change with the actual network.
