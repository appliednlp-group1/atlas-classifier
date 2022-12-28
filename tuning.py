import argparse
import os

import optuna

from train import run


def tuning(bert_model: str,
           contriever_model: str,
           contriever_path: str,
           dataset_path: str,
           index_path: str,
           batch_size: int,
           num_epochs: int,
           use_ratio: float,
           result_dir: str,
           trial_name: str,
           n_trials: int = 50):
    
    def objective(trial):
        trial_id = str(trial._trial_id)
        out_dir = os.path.join(result_dir, trial_name, trial_id)
        
        q_encoder_lr = trial.suggest_loguniform('q_encoder_lr', 1e-6, 1e-4)
        classifier_lr = trial.suggest_loguniform('classifier_lr', 1e-6, 1e-4)
        
        return run(bert_model,
                   contriever_model,
                   contriever_path,
                   dataset_path,
                   index_path,
                   batch_size,
                   num_epochs,
                   q_encoder_lr,
                   classifier_lr,
                   out_dir,
                   use_ratio)
    
    study = optuna.create_study(
        study_name=trial_name,
        storage='sqlite:///./sqlite3.db',
        load_if_exists=True,
        direction='maximize',
        )
    study.optimize(objective, n_trials=n_trials)
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model',
                        type=str,
                        default='bert-base-uncased')
    parser.add_argument('--contriever_model',
                        type=str,
                        default='facebook/contriever')
    parser.add_argument('--contriever_path',
                        type=str,
                        default='models/models/atlas/base/model.pth.tar')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='data/datasets/train')
    parser.add_argument('--index_path',
                        type=str,
                        default='data/datasets/agnews.faiss')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=50)
    parser.add_argument('--use_ratio',
                        type=float,
                        default=0.01)
    parser.add_argument('--result_dir',
                        type=str,
                        default='results/tuning')
    parser.add_argument('--name',
                        type=str,
                        default='test1')
    parser.add_argument('--n_trials',
                        type=int,
                        default=50)

    args = parser.parse_args()

    tuning(args.bert_model,
           args.contriever_model,
           args.contriever_path,
           args.dataset_path,
           args.index_path,
           args.batch_size,
           args.num_epochs,
           args.use_ratio,
           args.result_dir,
           args.name,
           args.n_trials,
           )
