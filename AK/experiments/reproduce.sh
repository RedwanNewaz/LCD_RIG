
all_results()
{
  declare -a Envs=("N17E073" "N43W080" "N45W123" "N47W124")
  declare -a Strategies=("random" "active" "myopic")
  declare -a Kernels=("rbf" "gibbs" "dkl" "ak")

  for seed in {0..9}; do
    for env in ${Envs[@]}; do
      for strategy in ${Strategies[@]}; do
        for kernel in ${Kernels[@]}; do
          echo $seed $env $strategy $kernel
#
          python main.py --config ./configs/$kernel.yaml --env-name $env --strategy $strategy --seed $seed > "./loginfo/${seed}/${env}/${strategy}/${kernel}.txt"
        done
      done
    done
  done
}


single_instance()
{
  seed=0
  env="N45W123"
  strategy="distributed"
  kernel="ak"

  folder="./loginfo/${seed}/${env}/${strategy}"

  if [ -d ${folder} ]; then
    echo "[+] removing " ${folder}
    rm -rf ${folder}
  fi

  mkdir -p ${folder}
  touch "${folder}/${kernel}.txt"

  python="/home/redwan/anaconda3/envs/rig/bin/python"
  $python main.py --config ./configs/$kernel.yaml --env-name $env --strategy $strategy --seed $seed > "${folder}/${kernel}.txt"
}

single_instance


  > "${folder}/${kernel}.txt"

