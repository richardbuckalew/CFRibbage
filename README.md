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

250 million hands trained. Here is 200M vs 250M:

![training_ss](https://user-images.githubusercontent.com/6075739/227777883-50a75653-df86-4ee7-a11e-15672c0c5a38.png)
![250m](https://user-images.githubusercontent.com/6075739/227777978-9660b1f1-2382-4e3f-8fa8-fb31e213b026.png)

At the 200M milestone, nearly 80% of dealt hands resulted in no change to the learned strategy, so I have updated the training code to skip training when both the dealer hand and the pone hand have converged. It's up to 250M now, and the zero rate is already back up near 70%. That is a very crude measure of convergence, since as I noted it was higher than this previously and training was still very much ongoing. But the other indicators definitely suggest that convergence is near. For one thing, the 70% zero rate is *on top of* the previous 80%. The two numbers are not fully independent --- I'm only skipping deals where *both* players' hands have converged (for now). But this means probably 90% of deals involve hands that have converged. 


That does not mean that 90% of *hands* have converged though, since hands are drawn from their actual distribution, which is discrete with 6 values across several orders of magnitude. There are some hands which have only been dealt a single-digit number of times in those 250M deals. Once the current training regime begins to bog (probably at 300M), it will be time to selectively target those especially rare hands.

I also have a working frontend now, at [CFRibbage.love](https://github.com/richardbuckalew/CFRibbage.love)), and a partial backend written with Genie.jl (CFRibbageWeb.jl), that includes my first-ever REST API. I have successfully used love2d to poll a remote server in order to choose a discard! This is a big milestone for the project.

