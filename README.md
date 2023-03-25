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

250 million hands trained. Here is a snapshot of the training progress after 200M hands:

![image](https://user-images.githubusercontent.com/6075739/210567253-9dabd626-35d7-4350-912e-33aa935a0e21.png)

Nearly 70% of dealt hands resulted in no change to the learned strategy, so I have updated the training code to skip training when both the dealer hand and the pone hand have converged. It's up to 250M now, and the zero rate is back to reasonable levels. After about 100M of hands in this training regime, I'll begin microtargeting the really rare hands that haven't seen a statistically significant number of deals yet. Based on the learning rate, I'm guessing 400M hands dealt is a reasonable endpoint.

I have a working frontend now, at [CFRibbage.love](https://github.com/richardbuckalew/CFRibbage.love)), and a partial backend written with Genie.jl (CFRibbageWeb.jl) but not uploaded to github yet, that includes my first-ever REST API. I have successfully used love2d to poll a remote server in order to choose a discard! This is a big milestone for the project.

