import os
import yaml
import torch
import threading
import numpy as np
import pandas as pd
from dqn import DQNAgent
from collections import deque
import matplotlib.pyplot as plt
from connect4Env import Connect4Env
from torch.utils.tensorboard import SummaryWriter
import argparse # For command-line config path

# --- Load YAML configuration ---
def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        exit()
    

class Trainer:
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- Environment Setup ---
        env_config = self.config['environment']
        self.env = Connect4Env(
            rows=env_config['rows'],
            cols=env_config['cols'],
            win_length=env_config['win_length']
        )

        # --- Agent Setup ---
        agent_config = self.config["agent"]
        self.agent = DQNAgent(
            board_shape=(self.env.rows, self.env.cols),
            n_actions=self.env.action_space.n,
            device=self.device,
            learning_rate=agent_config['learning_rate'],
            gamma=agent_config['gamma'],
            replay_buffer_capacity=agent_config['replay_buffer_capacity'],
            batch_size=agent_config['batch_size'],
            target_update_freq=agent_config['target_update_freq']
        )

        # --- Trainer Attributes ---
        trainer_config = self.config['trainer']
        self.epsilon_start = trainer_config['epsilon_start']
        self.epsilon_end = trainer_config['epsilon_end']
        self.epsilon_decay_episodes = trainer_config['epsilon_decay_episodes']
        self.epsilon = self.epsilon_start
        self.min_epsilon = self.epsilon_end

        self.writer = SummaryWriter(log_dir=trainer_config['log_dir'])
        self.episode = 0
        self.steps = 0

        if self.epsilon_decay_episodes > 0:
            # Calculate decay factor to reach epsilon_end from epsilon_start in N episodes
            self.epsilon_decay_factor = (self.epsilon_end / self.epsilon_start)**(1.0 / self.epsilon_decay_episodes)
        else:
            self.epsilon_decay_factor = 1.0 
        
        self.checkpoint_dir = trainer_config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history = {
            "episodes": [], "rewards": [], "episode_steps": [],
            "epsilon_values": [], "results": [],
            "training_loss": [], "loss_steps": []
        }

        self.save_freq = trainer_config['save_freq']
        self.print_freq = trainer_config['print_freq']
        
    
    def _calculate_epsilon(self):
        if self.episode < self.epsilon_decay_episodes:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_factor)
        else:
            self.epsilon = self.min_epsilon
        return self.epsilon


    def save_checkpoint(self, episode_num, wait=False):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode_num}.pth")
        buffer_to_save = []
        if hasattr(self.agent.replay_buffer, 'buffer'):
            for transition in self.agent.replay_buffer.buffer:
                s, a, r, ns, d = transition
                buffer_to_save.append((
                    torch.from_numpy(s.astype(np.float32)) if isinstance(s, np.ndarray) else s,
                    torch.tensor(a, dtype=torch.long),
                    torch.tensor(r, dtype=torch.float32),
                    torch.from_numpy(ns.astype(np.float32)) if isinstance(ns, np.ndarray) else ns,
                    torch.tensor(d, dtype=torch.bool)
                ))
        
        checkpoint_data = {
            "episode": episode_num,
            "steps": self.steps,
            "online_state_dict": self.agent.online_net.state_dict(),
            "target_state_dict": self.agent.target_net.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "replay_buffer": buffer_to_save,
            "training_history": self.history,
            "config": self.config # Save the config used for this checkpoint
        }
        save_thread = threading.Thread(target=torch.save, args=(checkpoint_data, checkpoint_path))
        save_thread.daemon = not wait
        save_thread.start()

        if wait:
            print("Waiting for final checkpoint to finish saving...")
            save_thread.join()


    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            return False
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.agent.online_net.load_state_dict(checkpoint["online_state_dict"])
            self.agent.target_net.load_state_dict(checkpoint["target_state_dict"])
            self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            self.episode = checkpoint.get("episode", 0)
            self.steps = checkpoint.get("steps", 0)
            # Use epsilon from config for decay calculation, but can restore saved epsilon if desired.
            # self.epsilon = checkpoint.get("epsilon", self.epsilon_start) 
            # Recalculate current epsilon based on loaded episode and config's decay schedule
            current_epsilon_at_load = self.epsilon_start
            if self.epsilon_decay_episodes > 0:
                for _ep_count in range(self.episode):
                    if _ep_count < self.epsilon_decay_episodes:
                        current_epsilon_at_load = max(self.min_epsilon, current_epsilon_at_load * self.epsilon_decay_factor)
                    else:
                        current_epsilon_at_load = self.min_epsilon
                        break
            self.epsilon = current_epsilon_at_load


            loaded_buffer_data = checkpoint.get("replay_buffer", [])
            self.agent.replay_buffer.buffer.clear()
            temp_buffer_list = []
            for transition_tensors in loaded_buffer_data:
                s_t, a_t, r_t, ns_t, d_t = transition_tensors
                temp_buffer_list.append((s_t.cpu().numpy(), a_t.item(), r_t.item(), ns_t.cpu().numpy(), d_t.item()))
            self.agent.replay_buffer.buffer.extend(temp_buffer_list)
            
            if "training_history" in checkpoint:
                self.history = checkpoint["training_history"]
            else:
                 self.history = {key: [] for key in ["episodes", "p1_rewards", "episode_steps", "epsilon_values", "p1_results", "training_loss", "loss_steps"]}

            print(f"Loaded checkpoint from {checkpoint_path}. Resuming from episode {self.episode + 1}, steps {self.steps}. Current epsilon: {self.epsilon:.4f}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            return False
    
    def save_model_play(self, model_filename="dqn_agent.pth"):
        """Saves only the online network's state_dict for fast loading and inference"""
        model_path = os.path.join(self.checkpoint_dir, model_filename)

        # We only need the learned weights of the student/online network for playing
        torch.save(self.agent.online_net.state_dict(), model_path)
        print(f"Model saved for inference purposes to {model_path}")


    def train_episode(self):
        obs = self.env.reset()
        done = False
        ep_reward = 0.0
        ep_steps_count = 0

        while not done:
            current_player = self.env.current_player
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                if not done:
                    print(f"Warning: Ep {self.episode}, Step {ep_steps_count}. No valid actions. Board:\n{obs}")
                    break
            
            action = self.agent.select_action(
                state_2d=obs,
                epsilon=self.epsilon,
                valid_actions=valid_actions
            )
            next_obs_for_new_player, reward_for_action, done, info = self.env.step(action)

            if current_player == 1:
                ep_reward += reward_for_action
            else:
                ep_reward -= reward_for_action
            
            self.agent.store_transition(obs, action, reward_for_action, next_obs_for_new_player, done)
            loss = self.agent.learn()
            if loss is not None:
                self.writer.add_scalar("Training/Loss_per_step", loss, self.steps)
                self.history["training_loss"].append(loss)
                self.history["loss_steps"].append(self.steps)
            
            obs = next_obs_for_new_player
            ep_steps_count += 1
            self.steps += 1
        
        if not done:
            ep_result = "incomplete"
        elif ep_reward > 0.5: # P1 won
            ep_result = "win"
        elif ep_reward < -0.5: # P1 lost
            ep_result = "loss"
        else: 
            ep_result = "draw"
        return ep_reward, ep_steps_count, ep_result
    

    def _update_and_log_history(self, ep_reward, ep_steps, ep_result, current_epsilon):
        self.history["episodes"].append(self.episode)
        self.history["rewards"].append(ep_reward)
        self.history["episode_steps"].append(ep_steps)
        self.history["epsilon_values"].append(current_epsilon)
        self.history["results"].append(ep_result)
        self.writer.add_scalar("Episodic/Reward", ep_reward, self.episode)
        self.writer.add_scalar("Episodic/Steps_per_Episode", ep_steps, self.episode)
        self.writer.add_scalar("Params/Epsilon", current_epsilon, self.episode)

        win = 1 if ep_result == "win" else 0
        loss = 1 if ep_result == "loss" else 0
        draw = 1 if ep_result == "draw" else 0

        # Logging
        self.writer.add_scalar("Episodic/Outcome_Win", win, self.episode)
        self.writer.add_scalar("Episodic/Outcome_Loss", loss, self.episode)
        self.writer.add_scalar("Episodic/Outcome_Draw", draw, self.episode)

    def train(self):
        num_total_episodes = self.config['training']['num_total_episodes']
        checkpoint_path_to_load = self.config['training']['checkpoint_path_to_load']

        if checkpoint_path_to_load and os.path.exists(checkpoint_path_to_load):
            self.load_checkpoint(checkpoint_path_to_load)
        else:
            self.epsilon = self.epsilon_start
        
        print(f"Starting training from episode {self.episode + 1} up to {num_total_episodes}")

        try:
            while self.episode < num_total_episodes:
                self.episode += 1
                current_epsilon = self._calculate_epsilon()

                ep_reward, ep_steps, ep_result = self.train_episode()
                self._update_and_log_history(ep_reward, ep_steps, ep_result, current_epsilon)

                if self.episode % self.print_freq == 0:
                    print(f"Ep: {self.episode}/{num_total_episodes} | Steps: {self.steps} | "
                          f"Res: {ep_result} (Rew {ep_reward:.2f})  | EpSteps: {ep_steps} | "
                          f"Eps: {current_epsilon:.4f}")
                
                if self.episode % self.save_freq == 0:
                    self.save_checkpoint(self.episode)
                    self.save_model_play(model_filename=f"model_inference_{self.episode}.pth")
                    print(f"Checkpoint saved for episode {self.episode}")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            print("Saving final checkpoint")
            self.save_model_play(model_filename=f"model_inference_{self.episode}.pth")
            self.save_checkpoint(self.episode, wait=True)
            print(f"Final checkpoint for episode {self.episode} saved. Total steps: {self.steps}.")
            self.writer.close()
            print("TensorBoard writer closed. Training finished.")
            exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a Connect4 DQN agent.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    config_data = load_config(args.config)
    trainer = Trainer(config=config_data)
    trainer.train()