# CFRibbage.jl
*A Hybrid Counterfactual Regret Minimization solver for 2-player Cribbage*

This is a learning project in several ways:
- I want to learn Julia.
- I want to get familiar with Machine Learning.
- I want to learn how to use GitHub.
- I want to beat my wife at Cribbage.

The end-result of this project will be a **database** (in a cross-platform format) plus a **solver** for playing optimally at all stages in a game of 2-player cribabge. 

I welcome comments and ideas from interested strangers, but be nice -- this is a learning project 😬

**CURRENT STATUS**

200 million hands trained. Getting marginal returns on computation now, as more than 80% of the hands dealt result in no change to the strategy profile -- yet, of the hands that *do* see meaningful training, there's a ways to go yet. the 75th percential is still changing by about 4% per deal, and the maximum delta still lives up very close to 80%. Still, that's meaningful progress!

![image](https://user-images.githubusercontent.com/6075739/210567253-9dabd626-35d7-4350-912e-33aa935a0e21.png)

Now that I've starting writing the frontend (see CFRibbage.love), training will pause while I build out the Genie server (I'll be hooking into a lot of the same functions / structures that training uses). Expecting a fully-working alpha build within a couple of weeks.

