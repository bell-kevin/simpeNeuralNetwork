<a name="readme-top"></a>

# 
This is an example of a very simple neural network (a 2-layer multilayer perceptron) written from scratch in pure Python (using only NumPy for matrix math). It learns the XOR function. The code includes:

    - A tiny dataset (XOR truth table)
    
    - Random weight initialization
    
    - A simple forward pass (using the sigmoid activation)
    
    - A simple backward pass (gradient descent)
    
    - Training loop and final prediction

  How it works

    1. Data: We use the XOR dataset, which is a classic simple dataset where the neural network must learn the XOR operation.
    
    2. Weights: We initialize two weight matrices (W1, W2) and two biases (b1, b2) randomly (and zeros for biases).
    
    3. Forward pass:
    
        - Compute z1=XW1+b1 then apply the sigmoid to get a1​.
        
        - Compute z2=a1W2+b2 then apply the sigmoid to get a2 (the prediction).
        
    4. Loss: We use mean squared error (MSE) for simplicity.
    
    5. Backward pass: We compute the gradients of the loss with respect to each parameter using the chain rule and then update the parameters with gradient descent.
    
    6. Training: After enough epochs, the network learns to output the correct XOR result for each input.

This example is intentionally small and simple, but it demonstrates the core concepts of neural networks (forward pass, loss computation, backward pass, and parameter updates).

--------------------------------------------------------------------------------------------------------------------------
== We're Using GitHub Under Protest ==

This project is currently hosted on GitHub.  This is not ideal; GitHub is a
proprietary, trade-secret system that is not Free and Open Souce Software
(FOSS).  We are deeply concerned about using a proprietary system like GitHub
to develop our FOSS project. I have a [website](https://bellKevin.me) where the
project contributors are actively discussing how we can move away from GitHub
in the long term.  We urge you to read about the [Give up GitHub](https://GiveUpGitHub.org) campaign 
from [the Software Freedom Conservancy](https://sfconservancy.org) to understand some of the reasons why GitHub is not 
a good place to host FOSS projects.

If you are a contributor who personally has already quit using GitHub, please
email me at **bellKevin@pm.me** for how to send us contributions without
using GitHub directly.

Any use of this project's code by GitHub Copilot, past or present, is done
without our permission.  We do not consent to GitHub's use of this project's
code in Copilot.

![Logo of the GiveUpGitHub campaign](https://sfconservancy.org/img/GiveUpGitHub.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
