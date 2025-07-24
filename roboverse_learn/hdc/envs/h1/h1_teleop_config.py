from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1TeleopCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        max_linvel = 0.5
        max_angvel = 0.5
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }
    
    class domain_rand ( LeggedRobotCfg.domain_rand ):
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.5
        
        randomize_friction = True
        # randomize_friction = False
        friction_range = [-0.6, 1.2]
        
        randomize_base_mass = False # replaced by randomize_link_mass
        added_mass_range = [-5., 10.]


        randomize_base_com = True
        class base_com_range: #kg
            x = [-0.1, 0.1]
            y = [-0.1, 0.1]
            z = [-0.1, 0.1]

        randomize_link_mass = True
        link_mass_range = [0.7, 1.3] # *factor
        randomize_link_body_names = [
            'pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 
            'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link',  'torso_link',
        ]

        randomize_pd_gain = True
        kp_range = [0.75, 1.25]
        kd_range = [0.75, 1.25]


        randomize_torque_rfi = True
        rfi_lim = 0.1
        randomize_rfi_lim = True
        rfi_lim_range = [0.5, 1.5]

        randomize_motion_ref_xyz = True
        motion_ref_xyz_range = [[-0.02, 0.02],[-0.02, 0.02],[-0.05, 0.05]]
        
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [1, 3] # integer max real delay is 90ms
        

    class noise ( LeggedRobotCfg.noise ):
        add_noise = True # False for teleop sim right now
        noise_level = 1.0 # scales other values
        class noise_scales:
            base_z = 0.05
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.2
            lin_acc = 0.2 # ???????????????
            ang_vel = 0.5
            gravity = 0.1
            in_contact = 0.1
            height_measurements = 0.05
            body_pos = 0.01 # body pos in cartesian space: 19x3
            body_lin_vel = 0.01 # body velocity in cartesian space: 19x3
            body_rot = 0.001 # 6D body rotation 
            delta_base_pos = 0.05
            delta_heading = 0.1
            last_action = 0.0
            
            ref_body_pos = 0.10
            ref_body_rot = 0.01
            ref_body_vel = 0.01
            ref_lin_vel = 0.01
            ref_ang_vel = 0.01
            ref_dof_pos = 0.01
            ref_dof_vel = 0.01
            ref_gravity = 0.01
    class sim ( LeggedRobotCfg.sim ):
        dt = 0.005  #   1/60.
    class terrain ( LeggedRobotCfg.terrain ):
        # mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        # mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        # border_size = 25 # [m] # for play only
        curriculum = False
        # curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        # measure_heights = False # keep it False
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measure_heights = False 
        measured_points_x = [ 0.] # 1mx1.6m rectangle (without center line)
        measured_points_y = [ 0.]

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 9 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        terrain_types = ['flat', 'rough', 'low_obst', 'smooth_slope', 'rough_slope']  # do not duplicate!
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        # terrain_proportions = [0.2, 0.6, 0.2, 0., 0.]
        # terrain_proportions = [1,, 0., 0., 0., 0.]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # num_observations = 88 # v-min2
        # num_privileged_obs = 164 # v-min2
        # num_observations = 624
        # num_observations = 87 # v-teleop
        # num_privileged_obs = 163 # v-teleop
        # num_observations = 84 # v-teleop-clean
        # num_privileged_obs = 160 # v-teleop-clean
        # num_observations = 75 # v-teleop-superclean
        # num_privileged_obs = 151 # v-teleop-superclean
        # num_observations = 65 # v-teleop-clean-nolastaction
        # num_privileged_obs = 141 # v-teleop-clean-nolastaction
        # num_observations = 90 # v-teleop_extend
        # num_privileged_obs = 166 # v-teleop_extend
        # num_observations = 87 # v-teleop_extend_nolinvel
        # num_privileged_obs = 163 # v-teleop_extend_nolinvel

        num_observations = 138 # v-teleop-extend-max
        num_privileged_obs = 214 # v-teleop-extend-max

        # num_observations = 135 # v-teleop-extend-max-nolinvel
        # num_privileged_obs = 211 # v-teleop-extend-max-nolinvel

        # num_observations = 138 # v-teleop-extend-max-acc
        # num_privileged_obs = 214 # v-teleop-extend-max-acc
        
        num_actions = 19
        im_eval = False
      
    class commands ( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 0.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [.0, .0] # min max [m/s]
            lin_vel_y = [.0, .0]   # min max [m/s]
            ang_vel_yaw = [.0, .0]    # min max [rad/s]
            heading = [.0, .0]
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # stiffness = {'hip_yaw': 200,
        #              'hip_roll': 200,
        #              'hip_pitch': 200,
        #              'knee': 300,
        #              'ankle': 40,
        #              'torso': 300,
        #              'shoulder': 100,
        #              "elbow":100,
        #              }  # [N*m/rad]
        # damping = {  'hip_yaw': 15,
        #              'hip_roll': 15,
        #              'hip_pitch': 15,
        #              'knee': 18,
        #              'ankle': 6,
        #              'torso': 18,
        #              'shoulder': 6,
        #              "elbow":6,
        #              }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4 # 4

        # action_filt = False
        action_filt = False
        action_cutfreq = 4.0

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["pelvis", "shoulder", "hip", "knee"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
  
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        set_dof_properties = True
        default_dof_prop_damping = [5,5,5,6,2, 5,5,5,6,2, 6, 2,2,2,2, 2,2,2,2]
        # default_dof_prop_stiffness = [200,200,200,300,40, 200,200,200,300,40, 300, 100,100,100,100, 100,100,100,100]
        default_dof_prop_stiffness = [0,0,0,0,0, 0,0,0,0,0, 0, 0,0,0,0, 0,0,0,0]
        default_dof_prop_friction = [0,0,0,0,0, 0,0,0,0,0, 0, 0,0,0,0, 0,0,0,0]
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        
        terminate_by_knee_distance = False
        terminate_by_low_height = True
        terminate_by_lin_vel = False
        terminate_by_ang_vel = False
        terminate_by_gravity = True
        terminate_by_low_height = False
        
        terminate_by_ref_motion_distance = True
        terminate_by_1time_motion = True
        
        local_upper_reward = False
        zero_out_far= False # Zero out far termination
        close_distance = 0.25
        far_distance =5
        
        class termination_scales():
            base_height = 0.3
            base_vel = 10.0
            base_ang_vel = 5.0
            gravity_x = 0.7
            gravity_y = 0.7
            min_knee_distance = 0.
            max_ref_motion_distance = 0.5


    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            torques = -9e-5*1.25
            torque_limits = -2e-1*1.25
            dof_acc = -8.4e-6*1.25 #-8.4e-6   -4.2e-7 #-3.5e-8
            dof_vel = -0.003*1.25 # -0.003
            # action_rate = -0.6 # -0.6  # -0.3 # -0.3 -0.12 -0.01
            lower_action_rate = -0.9*1.25 # -0.6  # -0.3 # -0.3 -0.12 -0.01
            upper_action_rate = -0.05*1.25 # -0.6  # -0.3 # -0.3 -0.12 -0.01
            dof_pos_limits = -100.0*1.25
            termination = -200*1.25
            feet_contact_forces = -0.10*1.25
            stumble = -1000.0*1.25
            feet_air_time_teleop = 800.0*1.25
            slippage = -30.0*1.25
            feet_ori = -50.0*1.25
            orientation = -0.0
            teleop_selected_joint_position = 32 # 5.0
            teleop_selected_joint_vel = 16 # 5.
            
            teleop_body_position = 0.0 # 6 keypoint
            teleop_body_position_extend = 40.0 # 8 keypoint
            teleop_body_position_extend_small_sigma = 0.0 # 8 keypoint
            teleop_body_rotation = 20.0
            teleop_body_vel = 8.0
            teleop_body_ang_vel = 8.0
            
            # slippage = -1.

        
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.85 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.85
        soft_torque_limit = 0.85

        max_contact_force = 500.

        teleop_joint_pos_sigma = 0.5
        teleop_joint_vel_sigma = 10.
        teleop_body_pos_sigma = 0.5 # 0.01
        teleop_body_pos_small_sigma = 0.01
        teleop_body_pos_lower_weight = 0.5
        teleop_body_pos_upper_weight = 1.0
        teleop_body_rot_sigma = 0.1
        teleop_body_vel_sigma = 10.
        teleop_body_ang_vel_sigma = 10.
        teleop_body_rot_selection = ['pelvis']
        teleop_body_vel_selection = ['pelvis']
        teleop_body_pos_selection = ['pelvis']
        teleop_body_ang_vel_selection = ['pelvis']
        teleop_joint_pos_selection = {
                # upper body
                'torso_joint': 2.0,
                'left_shoulder_pitch_joint': 2.0,
                'left_shoulder_roll_joint': 2.0,
                'left_shoulder_yaw_joint': 2.0,
                'left_elbow_joint': 2.0,
                'right_shoulder_pitch_joint': 2.0,
                'right_shoulder_roll_joint': 2.0,
                'right_shoulder_yaw_joint': 2.0,
                'right_elbow_joint': 2.0,
                # lower body
                'left_hip_pitch_joint': 2.0,
                'left_hip_roll_joint': 0.5,
                'left_hip_yaw_joint': 0.5,
                'left_knee_joint': 0.5,
                'left_ankle_joint': 0.5,
                'right_hip_pitch_joint': 2.0,
                'right_hip_roll_joint': 0.5,
                'right_hip_yaw_joint': 0.5,
                'right_knee_joint': 0.5,
                'right_ankle_joint': 0.5,
        }
              
            
    class normalization:
        class obs_scales: # no normalization for nows
            lin_vel = 1.0 # 2.0
            lin_acc = 1.0 # ????????????
            ang_vel = 1.0 # 0.25
            dof_pos = 1.0 # 1.0
            dof_vel = 1.0 # 0.05
            height_measurements = 1.0 # 5.0
            body_pos = 1.0
            body_lin_vel = 1.0
            body_rot = 1.0
            delta_base_pos = 1.0
            delta_heading = 1.0
        clip_observations = 100.
        clip_actions = 100.


    class motion (LeggedRobotCfg.motion):
        teleop = True
        recycle_motion = True
        terrain_level_down_distance = 0.5
        num_markers = 19
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/walking_gesture.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/standing_one_gesture.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/standing.pkl'\
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/standing_20s_fpaa30.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/stable_wave_short.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/stable_wave_short_fpaa30.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/standing_20s_fpaa30.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/wave_and_walk_unfiltered.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/amass_run.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/gestures_3.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/walking_gesture_filered_12_fix.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/walking_gesture_filtered_fix.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/walking_gesture_filered_4.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/stable_punch.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/stable_amass.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/walk_fitted.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/bent_slowalk.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/walking_gesture_17.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/amass_and_stable_phc_filtered.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/amass_phc_filtered.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/amass_phc_filtered_shrinked800.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/amass_full.pkl'
        # motion_file = "/hdd/zen/dev/copycat/h1_phc/data/h1/v2/singles/test.pkl"
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/amass_phc_clean.pkl'
        # motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/amass_phc_clean_smooth.pkl'

        motion_file = '{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/amass_phc_filtered.pkl'

        skeleton_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/xml/h1.xml'
        marker_file = '{LEGGED_GYM_ROOT_DIR}/resources/objects/Marker/traj_marker.urdf'
        num_dof_pos_reference = 19
        num_dof_vel_reference = 19

        curriculum = False
        teleop_level_up_episode_length = 150
        teleop_level_down_episode_length = 100

        
        # eleop_obs_version = 'v-teleop'
        # teleop_obs_version = 'v-teleop-clean'
        # teleop_obs_version = 'v-teleop-superclean'
        # teleop_obs_version = 'v-teleop-clean-nolastaction'
        # teleop_obs_version = 'v-teleop-extend'
        # teleop_obs_version = 'v-teleop-extend-nolinvel'
        teleop_obs_version = 'v-teleop-extend-max'
        # teleop_obs_version = 'v-teleop-extend-max-nolinvel'
        # teleop_obs_version = 'v-teleop-extend-max-acc'
        # teleop_obs_version = 'v-min2'
        teleop_selected_keypoints_names = ['left_ankle_link',  'right_ankle_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_elbow_link', 'right_elbow_link']
        

        resample_motions_for_envs = True
        resample_motions_for_envs_interval_s = 1000
        extend_head = False
        extend_hand = True
        
class H1TeleopCfgPPO( LeggedRobotCfgPPO ):
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 0.2
        action_smoothness_coef = 0.000 # 0.003
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'h1:teleop'
        max_iterations = 10000000
        
        
    class policy ( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        # actor_hidden_dims = [512*4, 256*4, 128*4]
        critic_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [512*4, 256*4, 128*4]
  
