---
abstract: Neural Radiance Fields (NeRFs) have recently emerged as a powerful paradigm for the representation of natural, complex 3D scenes. NeRFs represent continuous volumetric density and RGB values in a neural network, and generate photo-realistic images from unseen camera viewpoints through ray tracing.  We propose an algorithm for navigating a robot through a 3D environment represented as a NeRF using only an on-board RGB camera for localization.  We assume the NeRF for the scene has been pre-trained offline, and the robot's objective is to navigate through unoccupied space in the NeRF to reach a goal pose.  We introduce a trajectory optimization algorithm that avoids collisions with high-density regions in the NeRF based on a discrete time version of differential flatness that is amenable to constraining the robot's full pose and control inputs.  We also introduce an optimization based filtering method to estimate 6DoF pose and velocities for the robot in the NeRF given only an onboard RGB camera.  We combine the trajectory planner with the pose filter in an online replanning loop to give a vision-based robot navigation pipeline.  We present simulation results with a quadrotor robot navigating through a jungle gym environment, the inside of a church, and Stonehenge using only an RGB camera. We also demonstrate an omnidirectional ground robot navigating through the church, requiring it to reorient to fit through the narrow gap.
---
(cool videos etc etc.)

<div class="d-none d-md-block abstract">
	<h4> Abstract </h4>
	{{page.abstract}}
</div>
<a class="d-block d-md-none" data-toggle="collapse" data-target="#collapseExample" aria-expanded="false" aria-controls="collapseExample"><b>[<u>abstract</u>]</b></a>
<div class="collapse" id="collapseExample">
  <div class="card card-body abstract">
    {{page.abstract}}
  </div>
</div>


<h4> Cool Videos </h4>
etc etc.