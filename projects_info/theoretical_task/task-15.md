## SOC: Self-Organized Criticality in the Sandpile Model

Self-Organized Criticality is an approach adopted to explain the emergence of power-law distributions in natural phenomena. 
In physics, criticality is assiocated to systems near phase transitions, that live between two distinct regimes, 
and at this critical point, the emergence of power-laws occurs.

In this report, we analyze the SOC behavior associated to the Sandpile Model.

**Sandpile Model**:

We consider networks with an associated load variable, that accounts for any sort of stress that can be observed in real applications (such as power grids). A given node in the network stays functional, as long as its load doesn't go above a given threshold: its *capacity*. When the capacity of a node is overcame, we have a failure in the node, and its load gets redistributed according to some chosen rules.

In this work we will use different types of networks, and observe the emergence of power-law distribution for the size $A$ of the avalances: $p(A)\sim A^{-\tau}$.

Analysis:

**Paper:** [Catastrophic Cascade of Failures in Interdependent Networks (Buldyrev et al. 2010)](./theoretical_task/jpsj.64.327.pdf)

Create networks with edges distribution $P(K)$:

- Gaussian(<$k$>=$20$ , $\sigma = 8$) (k>0)!
- Uniform(k $_{min}4$, k $_{max}=36$)

Use $N = 20000$ total nodes, and $N_b =200$ ($1\%$) boundary nodes ($bn$), where the $bn$ nodes are randomly selected, with $k(i)-n(i) \ge 0 $ connections to other nodes. These boundary nodes dissipate the load through the $n(i)$ connections.

Run the sandpile dynamics with threshold $k(i)$, evaluate distribution of avalanche sizes and the lifetime distributions, compare them with the values expected from multiplicative branching processes theory. 
Plot in loglog both distributions, and compare them to the expected values for the exponents in the thermodynamics limit.

- **Paper:** [Cascading Failures in Complex Networks (Motter & Lai 2002)](./theoretical_task/PhysRevLett.91.148701.pdf)

Create networks with edges distribution $P(K)$:

- $P(k)\propto k^{-\gamma}$

with $\gamma\in$ {$2.01,2.2,2.4,2.6,2.8,3.0,5.0,\infty$}, $N = 10^6$ and <$k$>$=4$. 

We start form N indexed nodes, and connect them using $w_i=i^{-\alpha}$ where $\gamma = 1+ \alpha^{-1}$, sampling two nodes i,j from the distribution $p(i)=\frac{w_i}{\sum_k w_k}$ and adding an edge between them if it doesn't exist. Repeat this until the mean degree of the network is $4$.

Run the sandpile dynamics with threshold $k(i)$ and a probability of losing a grain $f=10^{-4}$, evaluate distribution of avalanche sizes and the lifetime distributions, compare them with the values expected from multiplicative branching processes theory. 
Plot in loglog both distributions, and compare them to the expected values for the exponents in the thermodynamics limit.