#!/bin/bash
#SBATCH --job-name=multinli
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1

#SBATCH --output=/scratch/yzhou31/projects/CS688/logs/gpu_job_multinli-%j.out
#SBATCH --error=/scratch/yzhou31/projects/CS688/logs/gpu_job_multinli-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --mem=100G
#SBATCH --time=1-23:59:00

#SBATCH --array=1
# Load needed modules
# module load <package>
source /home/yzhou31/pytorch-env/bin/activate

# Define an array of your script names
# scripts=("model1.py" "model2.py" "model3.py" "model4.py" "model5.py")"main1.py" "main2.py" "main3.py" "main4.py" "main5.py"
# scripts=("main1.py" "main2.py" "main3.py" "main4.py" "main5.py" "main6.py" "main7.py" "main8.py" "main9.py" "main10.py" "main11.py" "main12.py" "main13.py" "main14.py" "main15.py" "main16.py" "main17.py" "main18.py" "main19.py" "main20.py" "main21.py" "main22.py" "main23.py" "main24.py" "main25.py")
# scripts=("main1.py" "main6.py" "main11.py" "main16.py" "main21.py")
# scripts=("main26.py" "main27.py" "main28.py" "main29.py" "main30.py" "main31.py" "main32.py" "main33.py" "main34.py" "main35.py" "main36.py" "main37.py" "main38.py" "main39.py" "main40.py" "main41.py" "main42.py" "main43.py" "main44.py" "main45.py" "main46.py" "main47.py" "main48.py" "main49.py" "main50.py")
# scripts=("main6.py" "main7.py" "main8.py" "main9.py" "main10.py" "main11.py" "main12.py" "main13.py" "main14.py" "main15.py" "main16.py" "main17.py" "main18.py" "main19.py" "main20.py" "main21.py" "main22.py" "main23.py" "main24.py" "main25.py")
# scripts=("main6.py" "main7.py" "main8.py" "main9.py" "main10.py")
scripts=("main.py")

# Get the script name for this task
script=${scripts[$SLURM_ARRAY_TASK_ID-1]}

# Execute
python3 $script

# done
echo "Done"
