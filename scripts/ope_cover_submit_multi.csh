#!/bin/csh
# 
## To run, type:
#  ./ope_cover_submit_multi.csh
#
# Script must have execute permissions, i.e.,
#  chmod u+x ope_cover_submit_multi.csh

module load conda

set MAX_EPISODE_LEN = (25) 
set NUM_TRAJS = (500 1000)
set DROPOUT_RATE = (0.9)
set DISCOUNT = 0.8
set MC_SIZE = 250
set EVAL_POLICY_MC_SIZE = 10000
set BURN_IN = 0
set SCALE_STATE = False
set BOOTSTRAP = False # False, True
set DROPOUT_SCHEME = ('3.19' '3.19-mar') # ('3.19' '3.20' '3.19-mar' '3.20-mar')
set BASIS_TYPE = 'spline'
set DROPOUT_OBS_COUNT_THRES = 2

conda activate /usr/local/usrapps/statistics/hwang77/env_rl

foreach max_episode_length ($MAX_EPISODE_LEN)
	foreach num_trajs ($NUM_TRAJS)
		foreach dropout_rate ($DROPOUT_RATE)
			
			bsub -n 2 -R "span[hosts=1]" -W 24:00 \
			-o out-${num_trajs}-scheme0.%J -e err-${num_trajs}-scheme0.%J \
			"python ope_cover.py --max_episode_length ${max_episode_length} --discount ${DISCOUNT} --num_trajs ${num_trajs} --burn_in ${BURN_IN} --mc_size ${MC_SIZE} --eval_policy_mc_size ${EVAL_POLICY_MC_SIZE} --eval_horizon 250 --scale_state ${SCALE_STATE} --dropout_obs_count_thres ${DROPOUT_OBS_COUNT_THRES} --ipw False --dropout_scheme '0' --dropout_rate ${dropout_rate} --estimate_missing_prob False --bootstrap ${BOOTSTRAP} --basis_type ${BASIS_TYPE}"

			foreach scheme ($DROPOUT_SCHEME)
				
				bsub -n 2 -R "span[hosts=1]" -W 24:00 \
				-o out-${num_trajs}-scheme${scheme}.%J -e err-${num_trajs}-scheme${scheme}.%J \
				"python ope_cover.py --max_episode_length ${max_episode_length} --discount ${DISCOUNT} --num_trajs ${num_trajs} --burn_in ${BURN_IN} --mc_size ${MC_SIZE} --eval_policy_mc_size ${EVAL_POLICY_MC_SIZE} --eval_horizon 250 --scale_state ${SCALE_STATE} --dropout_obs_count_thres ${DROPOUT_OBS_COUNT_THRES} --ipw False --dropout_scheme ${scheme} --dropout_rate ${dropout_rate} --estimate_missing_prob False --bootstrap ${BOOTSTRAP} --basis_type ${BASIS_TYPE}"

				bsub -n 2 -R "span[hosts=1]" -W 24:00 \
				-o out-${num_trajs}-scheme${scheme}.%J -e err-${num_trajs}-scheme${scheme}.%J \
				"python ope_cover.py --max_episode_length ${max_episode_length} --discount ${DISCOUNT} --num_trajs ${num_trajs} --burn_in ${BURN_IN} --mc_size ${MC_SIZE} --eval_policy_mc_size ${EVAL_POLICY_MC_SIZE} --eval_horizon 250 --scale_state ${SCALE_STATE} --dropout_obs_count_thres ${DROPOUT_OBS_COUNT_THRES} --ipw True --dropout_scheme ${scheme} --dropout_rate ${dropout_rate} --estimate_missing_prob False --bootstrap ${BOOTSTRAP} --basis_type ${BASIS_TYPE}"

				bsub -n 2 -R "span[hosts=1]" -W 48:00 \
				-o out-${num_trajs}-scheme${scheme}.%J -e err-${num_trajs}-scheme${scheme}.%J \
				"python ope_cover.py --max_episode_length ${max_episode_length} --discount ${DISCOUNT} --num_trajs ${num_trajs} --burn_in ${BURN_IN} --mc_size ${MC_SIZE} --eval_policy_mc_size ${EVAL_POLICY_MC_SIZE} --eval_horizon 250 --scale_state ${SCALE_STATE} --dropout_obs_count_thres ${DROPOUT_OBS_COUNT_THRES} --ipw True --dropout_scheme ${scheme} --dropout_rate ${dropout_rate} --estimate_missing_prob True --bootstrap ${BOOTSTRAP} --basis_type ${BASIS_TYPE}"

			end
		end
	end
end
echo "Completed submitting all jobs."

conda deactivate