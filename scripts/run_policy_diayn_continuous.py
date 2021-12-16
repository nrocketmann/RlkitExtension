from rlkit.samplers.util import DIAYNRollout as rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.envs.wrappers import NormalizedBoxEnv
import gym
import torch
import argparse
#import joblib
import uuid
from rlkit.core import logger
import numpy as np
import os

filename = str(uuid.uuid4())


def simulate_policy(args):
 #   data = joblib.load(args.file)
    policy = torch.load(os.path.join(args.file,'evaluation','policy','params.pt'))

    # data = torch.load(args.file)
    #
    # policy = data['evaluation/policy']
    env = NormalizedBoxEnv(gym.make(args.env))
 #   env = env.wrapped_env.unwrapped
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    video_dir = os.path.join('videos',args.vidname)
    if os.path.exists(video_dir):
        raise FileExistsError("This video folder already exists")
    else:
        os.mkdir(video_dir)

    import cv2
    index = 0
    skill_dim = policy.stochastic_policy.skill_dim
    if args.num_skills==-1:
        num_skills = skill_dim
    else:
        num_skills = args.num_skills
    for idx in range(num_skills):
        video = cv2.VideoWriter(os.path.join(video_dir,'video' + str(idx) + '.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30,
                                (500, 500))
        if args.num_skills==-1:
            skill = np.zeros([skill_dim])
            skill[idx] = 1
        else:
            skill = np.random.vonmises(0,2,[skill_dim])


        for _ in range(3):
            path = rollout(
                env,
                policy,
                skill,
                max_path_length=args.H,
                render=True,
            )
            if hasattr(env, "log_diagnostics"):
                env.log_diagnostics([path])
            logger.dump_tabular()
            for _ in range(30):
                video.write(np.zeros([500,500,3]).astype(np.uint8))
            for i, img in enumerate(path['images']):
                # print(i)
                # print(img.shape)
                video.write(img[:,:,::-1].astype(np.uint8))
#                cv2.imwrite("frames/diayn_bipedal_walker_hardcore.avi/%06d.png" % index, img[:,:,::-1])
                index += 1

        video.release()
        print("wrote video {0}".format(skill))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, default='Ant-v2', help="Which environment to run?")
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--num-skills',type=int,default=10,help="How many skills to videotape")
    parser.add_argument('--vidname',type=str, default='',help='name of your video')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
