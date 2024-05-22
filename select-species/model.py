import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycaret.regression import setup, compare_models, tune_model, save_model, predict_model, plot_model
from typing import List, Optional
import re
import argparse

class SelectSpecies:
    def __init__(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame, selected_species: Optional[List[int]] = None, 
                 independent_variable: Optional[List[str]] = None, dependent_variable: str = 'X4_mean', 
                 is_summary: bool = False, log_transformation: bool = True, preprocess: bool = True) -> None:
        
        self.is_summary = is_summary
        self.log_transformation = log_transformation
        self.dependent_variable = dependent_variable
        self.df = processed_df if is_summary else raw_df
        self.categorical_cols = ['Climate', 'Texture', 'prediction_label'] if is_summary else ['prediction_label']
        # if self.is_summary is true, make self.preprocess false
        if self.is_summary:
            self.preprocess = False
        else:
            self.preprocess = preprocess
        
        if log_transformation:
            self.apply_log_transformation()
        
        self.selected_species = selected_species if selected_species is not None else self.df.prediction_label.unique()
        if selected_species is not None:
            self.df = self.drop_specific_species(selected_species)
            
        if preprocess:
            self.preprocess_data()
            
        if independent_variable is not None:
            assert all(col in self.df.columns for col in independent_variable), 'independent_variable is not in the columns of self.df'
            self.independent_variable = independent_variable
        else:
            self.independent_variable = self._default_independent_variables()
        
    def _default_independent_variables(self):
        excluded_cols = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean', 
                         'X4_sd', 'X11_sd', 'X18_sd', 'X26_sd', 'X50_sd', 'X3112_sd']
        return [col for col in self.df.columns if col not in excluded_cols]

    def _apply_log_transformation(self, col):
        self.df[col] = np.log1p(self.df[col])

    def apply_log_transformation(self):
        exclude_cols = ['X4_mean', 'prediction_label', 'prediction_score', 'Climate', 'Texture'] if self.is_summary else ['X4_mean', 'prediction_label', 'prediction_score']
        cols = [col for col in self.df.columns if col not in exclude_cols]
        for col in cols:
            self._apply_log_transformation(col)

    def drop_specific_species(self, species: List[int]):
        return self.df[self.df.prediction_label.isin(species)]

    def preprocess_data(self):
        assert not self.is_summary, 'is_summary is True, preprocessing not required.'
        
        def calculate_mean_and_drop(df, col_pattern, new_col_name):
            cols = [col for col in df.columns if col_pattern in col]
            df[new_col_name] = df[cols].mean(axis=1)
            return df.drop(columns=cols)
        
        def filter_columns(df, include_patterns, exclude_patterns=[]):
            included_cols = [col for col in df.columns if all(p in col for p in include_patterns)]
            return [col for col in included_cols if not any(ep in col for ep in exclude_patterns)]
        
        self.df = calculate_mean_and_drop(self.df, 'soc', 'soc_average')
        
        worldclim_patterns = ['WORLDCLIM']
        worldclim_cols = [col for col in self.df.columns if 'WORLDCLIM' in col]
        annual_patterns = ['annual']
        exclude_patterns = ['range']
        annual_worldclim_cols = filter_columns(self.df, worldclim_patterns + annual_patterns, exclude_patterns)

        self.df = self.df.drop(columns=[col for col in self.df.columns if col in worldclim_cols and col not in annual_worldclim_cols])
        
        modis_bands = ['band_01_', 'band_02_', 'band_05_']
        for band in modis_bands:
            self.df = calculate_mean_and_drop(self.df, band, f'MODIS_{band[-2:]}_mean')
        
        vod_patterns = ['VOD_C', 'VOD_X', 'VOD_Ku']
        for pattern in vod_patterns:
            self.df = calculate_mean_and_drop(self.df, pattern, f'{pattern}_mean')
        
        soil_pattern = 'SOIL'
        texture_patterns = ['clay_0.5', 'sand_0.5', 'silt_0.5']
        texture_cols = filter_columns(self.df, [soil_pattern], texture_patterns)
        self.df = self.df.drop(columns=[col for col in texture_cols if col not in texture_patterns])
        
        self.df = self.df.drop(columns=[col for col in self.df.columns if len(col) > 50])
        
        sd_cols = [col for col in self.df.columns if 'sd' in col]
        self.df = self.df.drop(columns=sd_cols)

    def get_experiment_name(self):
        experiment_name = f"species_model_{self.dependent_variable}"
        if self.preprocess:
            experiment_name += "_preprocessed"
        if self.log_transformation:
            experiment_name += "_log_transformed"
        if self.is_summary:
            experiment_name += "_summary"
        if self.selected_species is not None:
            species_str = "_".join(str(species) for species in self.selected_species)
            experiment_name += f"_species_{species_str}"
        if self.independent_variable is not None:
            independent_vars_str = "_".join(self.independent_variable)
            experiment_name += f"_independent_{independent_vars_str}"
        return experiment_name

    def modeling(self, exp_name: str):
        
        def clean_column_names(df):
            df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
            return df

        self.df = self.df[self.independent_variable + [self.dependent_variable]]
        df = clean_column_names(self.df)
        
        reg = setup(data=df, target=self.dependent_variable, session_id=123, feature_selection=True, 
                    experiment_name=exp_name)
        
        best_model = compare_models()
        tuned_model = tune_model(best_model)
        
        return tuned_model
    
    def save_model(self, model):
        save_model(model, '/root/model')
    
    def prediction(self, model, test_data):
        predictions = predict_model(model, data=test_data)
        print(predictions)
        return predictions
    
    def plot_distribution(self, model):
        plot = plot_model(model, plot='feature', save=True)
        plt.savefig('/root/plot/feature_importance_plot.png')

def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for SelectSpecies")

    # 필수 인수
    parser.add_argument('--raw_df_path', type=str, required=True, help="Path to the raw dataframe CSV file")
    parser.add_argument('--processed_df_path', type=str, required=True, help="Path to the processed dataframe CSV file")
    parser.add_argument('--dependent_variable', type=str, required=True, help="Dependent variable")

    # 선택적 인수
    parser.add_argument('--log_transformation', type=bool, default=False, help="Whether to do log transformation")
    parser.add_argument('--selected_species', type=int, nargs='*', default=None, help="List of selected species (Integer list)")
    parser.add_argument('--is_summary', type=bool, default=False, help="Whether to use summary data")
    parser.add_argument('--preprocess', type=bool, default=True, help="Whether to preprocess the data")
    parser.add_argument('--independent_variable', type=str, nargs='*', default=None, help="List of independent variables")

    args = parser.parse_args()
    
    # Load dataframes
    raw_df = pd.read_csv(args.raw_df_path)
    processed_df = pd.read_csv(args.processed_df_path)

    return {
        'raw_df': raw_df,
        'processed_df': processed_df,
        'selected_species': args.selected_species,
        'independent_variable': args.independent_variable,
        'dependent_variable': args.dependent_variable,
        'is_summary': args.is_summary,
        'preprocess': args.preprocess,
        'log_transformation': args.log_transformation,
    }

if __name__ == "__main__":
    args = parse_args()
    plot_dist = SelectSpecies(**args)
    exp_name = plot_dist.get_experiment_name()
    print(exp_name)
    model = plot_dist.modeling(exp_name)
    plot_dist.save_model(model)
    plot_dist.plot_distribution(model)