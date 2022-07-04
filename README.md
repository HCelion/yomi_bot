## Yomi Bot

This is an attempts at an implementation of an asymmetric Rock-Paper-Scissor simulator that can learn to play optimal.
The inspiration for this asymmetric type of game is [Yomi](https://boardgamegeek.com/boardgame/43022/yomi).

A deeper dive into the principles can be found in the `/docs` folder.

The application makes use of the gambit library. You can download the version [here](https://sourceforge.net/projects/gambit/files/gambit16/16.0.1/) or a later one.

For Unix systems, unpack the file and then run inside the unpacked package
```
./configure
make
sudo make install
```

We also want to implement the game Yomi itself.
