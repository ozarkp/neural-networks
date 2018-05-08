# neural-networks
Implementations of various neural networks from the CSCI 252: Neural Networks (and Graphical Models) course during the Fall of 2016 at Washington and Lee University

## Example usage
Excluding the optical_flow-image_convolution example, all neural networks should be used with Python3. Open-CV does not work in Python3 without extra work.

### hopfield.py
	python3 hopfield.py

	Part 2: Vector-cosine confusion matrix of an array with itself ------------
	
	1.00
	0.62 1.00 
	0.45 0.33 1.00
	0.52 0.60 0.44 1.00
	0.58 0.40 0.50 0.56 1.00
  
	Part 3: Confusion matrix with 25 percent noise ------------
	
	0.69 
	0.50 0.84
	0.22 0.23 0.79
	0.65 0.55 0.52 0.80
	0.50 0.58 0.61 0.55 0.73
  
	Part 4: Recovering small patterns with a Hopfield net ------------
	
	Recover pattern, no noise:
	Input:  [0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 0 0 0 0 1 0 0 1 1 0 0 1 1 0]
	Output:  [0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 0 0 0 0 1 0 0 1 1 0 0 1 1 0]
	Vector cosine:  1.0
  
	Recover pattern, 25% noise:
	Input:  [0 1 1 0 1 0 0 1 0 0 1 1 0 1 1 0 0 0 0 0 1 0 1 1 0 1 0 1 1 0]
	Output:  [0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 0 0 0 0 1 0 0 1 1 0 0 1 1 0]
	Original:  [0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 0 0 0 0 1 0 0 1 1 0 0 1 1 0]
	Vector cosine:  1.0
  
	Part 5: Recovering big patterns ------------
	
	Confusion matrix for 1000-element vectors with 25% noise:
	0.74 
	0.52 0.74 
	0.49 0.47 0.73 
	0.50 0.50 0.50 0.76 
	0.49 0.49 0.51 0.47 0.73 
	0.50 0.48 0.48 0.49 0.46 0.74 
	0.52 0.49 0.53 0.50 0.50 0.47 0.76 
	0.48 0.50 0.49 0.52 0.50 0.50 0.54 0.76 
	0.48 0.49 0.49 0.51 0.46 0.52 0.50 0.49 0.78 
	0.46 0.47 0.49 0.49 0.46 0.47 0.49 0.50 0.52 0.74
  
	Recovering patterns with 25% noise:
	Vector cosine on pattern  0 =  1.0
	Vector cosine on pattern  1 =  1.0
	Vector cosine on pattern  2 =  1.0
	Vector cosine on pattern  3 =  1.0
	Vector cosine on pattern  4 =  1.0
	Vector cosine on pattern  5 =  1.0
	Vector cosine on pattern  6 =  1.0
	Vector cosine on pattern  7 =  1.0
	Vector cosine on pattern  8 =  1.0
	Vector cosine on pattern  9 =  1.0
	

### backprop.py
	python3 backprop.py
	
	...
	Iterative training data
	...
	
	Test on learning XOR: 
	[0 0] [ 0.01314943]
	[0 1] [ 0.97635649]
	[1 0] [ 0.97160169]
	[1 1] [ 0.0309646]
	

### lsa.py
	python3 lsa.py
	
	Step 1: Construct term-document matrix A and query matrix q:
	
 	              d1      d2      d3               q 

	a              1       1       1                0
	arrived        0       1       1                0
	damaged        1       0       0                0
	delivery       0       1       0                0
	fire           1       0       0                0
	gold           1       0       1                1
	in             1       1       1                0
	of             1       1       1                0
	shipment       1       0       1                0
	silver         0       2       0                1
	truck          0       1       1                1


	Step 2: Decompose matrix A into U, S, V^T:
	
	U:
 	[[-0.4201 -0.0748 -0.046 ]
 	[-0.2995  0.2001  0.4078]
 	[-0.1206 -0.2749 -0.4538]
 	[-0.1576  0.3046 -0.2006]
 	[-0.1206 -0.2749 -0.4538]
 	[-0.2626 -0.3794  0.1547]
 	[-0.4201 -0.0748 -0.046 ]
 	[-0.4201 -0.0748 -0.046 ]
 	[-0.2626 -0.3794  0.1547]
 	[-0.3151  0.6093 -0.4013]
 	[-0.2995  0.2001  0.4078]] 

	S:
 	[[ 4.0989  0.      0.    ]
 	[ 0.      2.3616  0.    ]
 	[ 0.      0.      1.2737]] 

	V^T:
 	[[-0.4945 -0.6458 -0.5817]
 	[-0.6492  0.7194 -0.2469]
 	[-0.578  -0.2556  0.775 ]] 



	Step 3: Implement a Rank 2 Approximation by keepingthe first two columns of U and V and the first two columns and rows of S:
	
	U_k:
 	[[-0.4201 -0.0748]
 	[-0.2995  0.2001]
 	[-0.1206 -0.2749]
 	[-0.1576  0.3046]
 	[-0.1206 -0.2749]
 	[-0.2626 -0.3794]
 	[-0.4201 -0.0748]
 	[-0.4201 -0.0748]
 	[-0.2626 -0.3794]
 	[-0.3151  0.6093]
 	[-0.2995  0.2001]] 

	S_k:
 	[[ 4.0989  0.    ]
 	[ 0.      2.3616]] 

	V_k^T:
 	[[-0.4945 -0.6458 -0.5817]
 	[-0.6492  0.7194 -0.2469]] 


	Step 4: Find the new document vector coordinates in this reduced 2-dimensional space. Rows of V holds eigenvector values. 		These are the coordinates of individual document vectors, hence:
	
	V:
 	[[-0.4945 -0.6492]
 	[-0.6458  0.7194]
 	[-0.5817 -0.2469]]


	Step 5: Find the new query vector coordinates in the reduced 2-dimensional space:
	
	Sk^-1:
 	[[ 0.244   0.    ]
 	[ 0.      0.4234]] 

	q:
 	[-0.214   0.1821]

	Step 6: Rank documents in decreasing order of query-document cosine similarities:

	sim(q, d1) =  -0.053951
	sim(q, d2) =  0.990987
	sim(q, d3) =  0.447959
	
### sdm.py
##### Must have matplotlib installed for Python3 (done easily with the 'pip3 install matplotlib' command
	Part 4: Recover pattern afer 25% noise added

	  *       * * * * * * *         
	*     * * * * * *     * *   *   
	* * * * *           * * * * *   
	  * * * *       *     * * *     
	* * * *   *         *     * *   
	* * *         *   * *     * * * 
	* *                 *   * * * * 
	* *     *   *     * *     *   * 
	  * *               *     *     
	  * * *   *     *   *       * * 
	* * * *   *     *         * * * 
	    * *             *   *   * * 
	  * *         *   * * * * *     
	*   *     *       * *   *       
	  *         * *   * * * * *     
	*             * * * *     * *   


	          * * * * * *           
	      * * * * * * * * * *       
	    * * * *         * * * *     
	  * * * *             * * * *   
	  * * *                 * * *   
	* * *                     * * * 
	* * *                     * * * 
	* * *                     * * * 
	* * *                     * * * 
	* * *                     * * * 
	* * *                     * * * 
	  * * *                 * * *   
	  * * * *             * * * *   
	    * * * *         * * * *     
	      * * * * * * * * * *       
	          * * * * * *           


	Part 5: Learn with the following five noisy examples: 

	          * * * * * *           
	      * * * * * * * * *         
	    * * * * *       * * * *   * 
	  * * * *             * * * *   
	  * * *         *   *   * * *   
	* * *   *                 * * * 
	* * *                     * * * 
	* * *                       * * 
	* * *               *     * * * 
	* * *   * *   *           * * * 
	* * *               *     * *   
	  * * * *                 * *   
	  * * * *         * * * * *     
	    * * * *         *   * *     
	      * * *   * * * * *         
	*     *   * *   *   *         * 


	    *     * * *   * *           
	  *   * * * * * * * * * *       
	    * * * *     *   * * * *     
	* * * * *   *           *   *   
	  * * *                 * *     
	*   *                     *     
	* * *               *     * * * 
	* * *   *                 * * * 
	* *                       * * * 
	* * *     *               * * * 
	* * *                     * * * 
	  * *               *   * * *   
	  * * * *             * * * *   
	    * * * *       * * * * *     
	      * * * * * * *   *         
	            * * * * *           


	*         * * * * *     *       
	*   * * * * * * * * * * * *   * 
	    * *   *         * *   *     
	  * * * *         *   * * * *   
	  *   *                 * * *   
	* *     *             *   * * * 
	  * *                     * * * 
	* * *                   * * *   
	*   *         *           * * * 
	* * *             *       * *   
	* *   *           * *     * * * 
	  * * *                 *   *   
	  * *   *             *   * *   
	    * * * *         * * * *     
	        * * * * * * * * *       
	*         * * * * * *           


	          * * * * * *           
	      * *   * * * * * * *       
	    * * *           * * * *     
	  * * *                 * * *   
	* * * *       *         * * *   
	* *   *     *             * * * 
	* * * *                   * * * 
	  * *                     * * * 
	* * *                     * * * 
	* * *             *       * * * 
	* * *                   * * * * 
	  * * *         *       * * *   
	  * * * *             * * * *   
	    *   * *     *     * * * *   
	      * * * * *   * * * *       
	      *   * * * * * *           


	            * *   * * *         
	    * * * * * * * * *   * *     
	*   * * * *         * *   *     
	  * * *               * * *   * 
	  * * *                 * * *   
	  * *                     * * * 
	*                         * *   
	* * *                     * * * 
	* * *   *     *     * *   * * * 
	* * *                     *   * 
	* * *   *                   * * 
	* * * *                   * *   
	  *   *     * *       * * * *   
	    * * *         * * * * *     
	      * * * * * * * * * *       
	          * * * * * * *         


	Test with the following probe: 

	          * * * * * *     *     
	      * * * * * * * * *         
	      * * *     *   * * * *     
	    * *               * * * * * 
	  * * *           *     * * *   
	* * *                 *   *   * 
	* * *           *         * * * 
	* * *                     * * * 
	* * *       * *         * * * * 
	*   * *                 * * * * 
	* * *           *           * * 
	* * * *                 * * *   
	  * *   *             * * * *   
	*   * * * *         * * *       
	        *   * * * * * * *       
	          * * *   * *           


	Result: 

	          * * * * * *           
	      * * * * * * * * * *       
	    * * * *         * * * *     
	  * * * *             * * * *   
	  * * *                 * * *   
	* * *                     * * * 
	* * *                     * * * 
	* * *                     * * * 
	* * *                     * * * 
	* * *                     * * * 
	* * *                     * * * 
	  * * *                 * * *   
	  * * * *             * * * *   
	    * * * *         * * * *     
	      * * * * * * * * * *       
	          * * * * * *        
						
### som.py
	python3 som.py

	Evaluating initial conditions of square...
<img width="628" alt="som_square_init" src="https://user-images.githubusercontent.com/9918239/39738642-f04ab432-5251-11e8-91f0-3ff6e7a7a81e.png">

	Evaluating final conditions of square...
<img width="632" alt="som_square_final" src="https://user-images.githubusercontent.com/9918239/39738681-186b4896-5252-11e8-87fb-e9ccafddf511.png">

	Evaluating initial conditions of ring...
<img width="625" alt="som_ring_init" src="https://user-images.githubusercontent.com/9918239/39738686-1cd2712a-5252-11e8-8526-613a35ad63f7.png">

	Evaluating final conditions of ring...
<img width="620" alt="some_ring_final" src="https://user-images.githubusercontent.com/9918239/39738689-1f70e1e6-5252-11e8-8f72-570d677f5194.png">
