# Generate each agents copy of the NN model 

initial_model = FourierNet()

models = [copy(initial_model) for i in range(N)]

# Get each nodes local data

datasets = [get_data(i) for i in range(N)]

# initialize the communication graph

G = generate_comm_graph(N)

# initialize the duals at zero

duals = [zeros_like(models[i]) for i in range(N)]

# main optimization loop
for _ in range(number_of_iterations):
    # Snapshot the current weights (communication)
    models_copy = copy(models.detach())

    # For all agents in parrallel
    for i in range(N):
    	# Determine neighbors for current iteration
    	neighs = G.get_neighbors(i)

    	# Get snapshots from neighbors (communication)
    	neighbor_models = models_copy[neighs]

    	# Update dual variable for i
    	duals[i] += rho * sum(models_copy[i] - neighbor_models)

    	# Compute the consensus regularizing terms for i
    	consensus_regs = (models_copy[i] + neighbor_models) / 2

    	# Initialize an optimizer for the primal step
    	opt = Adam(models[i], lr=alpha)

    	# Perform the approximate primal
    	for _ in range(B):
    	    opt.zero_grads()

    	    # Compute the local NeRF loss on a single batch
    	    local_loss = nerf_loss_single_batch(models[i], datasets[i])

    	    # Compute the dual loss component
    	    dual_loss = duals[i].T @ models[i]

    	    # Compute consensus regularization loss
    	    consensus_loss = rho * sum(square(cdist(models[i], consensus_regs)))

    	    # Compute full primal loss
    	    primal_loss = local_loss + dual_loss + consensus_loss

    	    # Regular backprop
    	    primal_loss.backward()
    	    opt.step()