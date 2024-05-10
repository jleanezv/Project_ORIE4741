import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

def convert_object_to_string(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                df[column] = df[column].astype('string')
            except Exception as e:
                print(f"Column {column} cannot be converted to string. Error: {e}")
    return df

def aggregate_df(df, interval):
    """
    Aggregates the 'value' column of a DataFrame that is indexed by time, 
    using the specified calculation_type ('CUMULATIVE', 'DISCRETE', 'MIN').
    It retains the last 'timezone' entry for each aggregated interval.
    
    Parameters:
    df (DataFrame): The dataframe with a DateTime index to aggregate.
    calculation_type (str): The type of calculation to perform. It should be one of the following:
                            'CUMULATIVE' for sum,
                            'DISCRETE' for mean,
                            'MIN' for minimum.
    interval (str): The time interval for aggregation ('T', 'H', '12H', 'D', 'W', 'M', '6M', 'Y').
    
    Returns:
    DataFrame: The aggregated dataframe with the specified calculation applied to 'value' and
               the last 'timezone' for each interval.
    """
    calculation_type = df['calculation_type'].iloc[0]
    
    calculations = {
        'CUMULATIVE': 'sum',
        'DISCRETE': 'mean',
        'MIN': 'min',
    }
    
    aggregation_functions = {
        'value': calculations[calculation_type], 
        'timezone': 'last'                      
    }

    aggregated_df = df.resample(interval).agg(aggregation_functions)
    
    return aggregated_df

def fill_value(df: DataFrame):
    """
    Returns a new DataFrame with the 'value' column filled with some imputation method.
    The imputation method is as follows:
    1. Convert 'value' to numeric.
    2. Replace 0 with NaN if 'timezone' is NaN.
    3. Interpolate linearly for NaN values.
    4. Backfill NaN values.
    5. Forward fill NaN values.
    """
    col = ['value']
    # df.loc[:,cols] = df.loc[:,cols].ffill()
    new_df = df.copy()
    new_df['value'] = pd.to_numeric(new_df['value'], errors='coerce').astype(float)
    mask = (new_df['value'] == 0) & (new_df['timezone'].isna())
    new_df.loc[mask, 'value'] = np.nan
    new_df.loc[:, col] = new_df.loc[:, col].interpolate(method='linear')
    new_df.loc[:, col] = new_df.loc[:, col].bfill()
    new_df.loc[:, col] = new_df.loc[:, col].ffill()
    return new_df

def display_data(df, first10=False, interpolate=True):
    # df = df.head(10).copy() if first10 else df.copy()
    # df['value'] = df['value'].interpolate(method='linear') if interpolate else df['value']

    plt.figure(figsize=(15,6))
    plt.plot(df.index, df['value'], marker='o')
    plt.title('Date vs Value')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.xticks(rotation=45) # Rotate the x-axis labels for better readability
    plt.tight_layout()
    plt.show()

def get_classifier_df(metric, user_dict):
    metric_df = pd.DataFrame()
    if metric in user_dict:
        metric_df = user_dict[metric].copy()
        metric_df[metric] = metric_df['value']
        metric_df.drop(columns=['value'], inplace=True)
    else:
        raise ValueError(f"Metric {metric} not found in user_dict")
    for contributor in contributor_dictionary[metric]:
        if contributor == metric:
            continue
        if contributor in user_dict:
            metric_df[contributor] = user_dict[contributor]['value']
    contributor_columns = [col for col in contributor_dictionary[metric] if col in metric_df.columns]
    metric_df = spread_values(metric_df)
    metric_df = metric_df[contributor_columns]
    if metric_df[metric].isnull().mean() > .6:
        raise ValueError(f"Primary metric {metric} has more than 60% missing values")
    metric_df = metric_df.loc[:, metric_df.isnull().mean() <= .6]
    metric_df[f'{metric}_tomorrow'] = metric_df[metric].shift(-1)
    metric_df = metric_df.dropna(subset=[f'{metric}_tomorrow'])
    metric_df[f'{metric}_mean'] = metric_df[metric].rolling(window=7, min_periods=1).mean()
    if metric == 'heartRateVariabilitySDNN' or metric == 'vO2Max' or metric == 'sleepEfficiency' or metric == 'timeNightDeep' or metric == 'timeNightRem':
        metric_df[f'{metric}_target'] = np.where(metric_df[f'{metric}_tomorrow'] >= metric_df[f'{metric}_mean'], 1, 0)
    elif metric == 'restingHeartRate':
        metric_df[f'{metric}_target'] = np.where(metric_df[f'{metric}_tomorrow'] <= metric_df[f'{metric}_mean'], 1, 0)
    else:
        print(f"Unknown metric: {metric}")
    
    return metric_df

def spread_values_algo(df, val, spread_freq, interpolate_freq):
    df_copy = df.copy()
    non_nan_idx = df_copy[val].notna()
    
    for idx in df_copy[non_nan_idx].index:
        current_value = df_copy.loc[idx, val]
        
        spread_timedelta = pd.Timedelta(spread_freq, unit='D')  # Assuming spread frequency is in days, adjust as needed
        
        start_idx = idx - spread_timedelta
        end_idx = idx
        if start_idx < df_copy.index[0]:
            start_idx = df_copy.index[0]
        df_copy.loc[start_idx:end_idx, val] = df_copy.loc[start_idx:end_idx, val].fillna(current_value)
        
        start_idx = idx
        end_idx = idx + spread_timedelta
        if end_idx > df_copy.index[-1]:
            end_idx = df_copy.index[-1]
        df_copy.loc[start_idx:end_idx, val] = df_copy.loc[start_idx:end_idx, val].fillna(current_value)
    
    gap_mask = (df_copy[val].notnull() != df_copy[val].shift().notnull()).cumsum()
    gap_size = gap_mask.map(gap_mask.value_counts())
    mask = (gap_size <= interpolate_freq) & (gap_size > 1)
    df_copy.loc[mask, val] = df_copy.loc[mask, val].interpolate(method='linear')
    
    return df_copy

metric_frequency_spread = {
    'heartRateVariabilitySDNN': (1, 3),
    'dietaryPotassium': (3, 7),
    'dietaryFatSaturated': (3, 7),
    'sleepEfficiency': (3, 7),
    'appleWalkingSteadiness': (7, 30),
    'dietaryNiacin': (3, 7),
    'walkingSpeed': (30, 7),
    'peakExpiratoryFlowRate': (30, 30),
    'dietaryProtein': (3, 7),
    'dietarySodium': (3, 7),
    'dietaryCholesterol':(3, 7),
    'timeNightInBed': (3, 7),
    'dietaryVitaminB6': (3, 7),
    'walkingHeartRateAverage': (7, 30),
    'oxygenSaturation': (7, 30),
    'heartRateVariabilityRMSSD': (1, 3),
    'dietaryCarbohydrates': (3, 7),
    'dietaryVitaminB12': (3, 7),
    'dietaryWater': (3, 7),
    'dietaryCalcium': (3, 7),
    'dietaryEnergyConsumed': (3, 7),
    'percentageSleepDeep': (3, 7),
    'dietaryFatTotal': (3, 7),
    'heartRateRecoveryOneMinute': (7, 30),
    'bodyMass': (15, 30),
    'leanBodyMass': (15, 30),
    'dietarySelenium': (3, 7),
    'timeSleepAwake': (3, 7),
    'basalEnergyBurned': (1, 3),
    'percentageNightDeep': (3, 7),
    'dietaryFolate': (3, 7),
    'bodyMassIndex': (15, 30),
    'dietaryVitaminC': (3, 7),
    'dietaryCaffeine': (3, 7),
    'respiratoryRate': (1, 3),
    'timeNightDeep': (3, 7),
    'percentageNightRem': (3, 7),
    'toothbrushMinutes': (1, 3),
    'bodyFatPercentage': (15, 30),
    'appleExerciseTime': (0, 2),
    'dietaryMagnesium': (3, 7),
    'percentageNightAwake': (3, 7),
    'dietaryPantothenicAcid':(3, 7),
    'dietaryCopper': (3, 7),
    'restingHeartRate': (1, 3),
    'dietaryPhosphorus': (3, 7),
    'dietaryVitaminD': (3, 7),
    'wakeupsOver15Minutes': (3, 7),
    'heartRate': (1, 3),
    'dietarySugar': (3, 7),
    'percentageSleepRem': (3, 7),
    'dietaryVitaminA': (3, 7),
    'percentageSleepLight': (3, 7),
    'dietaryVitaminE': (3, 7),
    'dietaryThiamin': (3, 7),
    'dietaryFatPolyunsaturated': (3, 7),
    'vO2Max': (7, 30),
    'bloodGlucose': (1, 3),
    'sleepWakeups': (3, 7),
    'sleepLatency': (3, 7),
    'dietaryZinc': (3, 7),
    'timeNightLight': (3, 7),
    'timeNightAwake': (3, 7),
    'timeNightRem': (3, 7),
    'percentageNightLight': (3, 7),
    'bloodPressureSystolic': (1, 3),
    'timeNightAsleep': (3, 7),
    'uVExposure': (3, 7),
    'forcedExpiratoryVolume1': (30, 30),
    'dietaryFatMonounsaturated': (3, 7),
    'walkingStepLength': (7, 30),
    'timeInDaylight': (3, 7),
    'walkingDoubleSupportPercentage': (7, 30),
    'bloodPressureDiastolic': (1, 3),
    'percentageSleepAwake': (3, 7),
    'dietaryIron': (3, 7),
    'dietaryVitaminK': (3, 7),
    'dietaryRiboflavin': (3, 7),
    'dietaryManganese': (3, 7),
    'dietaryFiber': (3, 7),
}

def spread_values(df):
    for col in df.columns:
        if col in metric_frequency_spread:
            spread_freq, interpolate_freq = metric_frequency_spread[col]
            df = spread_values_algo(df, col, spread_freq, interpolate_freq)
    return df

contributor_dictionary = {
    'heartRateVariabilitySDNN': [
        'heartRateVariabilitySDNN', 'distanceSwimming',
        'dietaryVitaminE', 'dietarySugar',
       'appleStandTime', 'dietaryFatPolyunsaturated', 'bloodPressureDiastolic',
       'restingHeartRate', 'timeNightDeep', 'bodyMass',
        'dietaryVitaminB12',
       'sleepEfficiency', 'timeInDaylight', 'bloodPressureSystolic',
       'dietaryWater', 'stepCount', 'dietaryPotassium', 'bloodGlucose',
       'dietaryCaffeine', 'dietaryVitaminD', 'dietaryNiacin',
       'appleExerciseTime', 'dietaryCarbohydrates', 'dietaryFatSaturated',
       'dietaryVitaminK', 'wakeupsOver15Minutes',
       'dietaryThiamin', 'mindfulMinutes',
        'environmentalAudioExposure', 'dietaryCholesterol',
       'dietaryCopper', 'dietaryVitaminB6', 'forcedExpiratoryVolume1',
       'dietaryVitaminA', 'bodyMassIndex', 'dietaryEnergyConsumed', 'vO2Max',
       'dietaryZinc', 'headphoneAudioExposure', 'dietaryMagnesium', 'dietaryFatTotal', 'timeSleepAwake',
       'dietaryFatMonounsaturated', 'dietaryPantothenicAcid', 'dietaryProtein',
        'dietaryCalcium', 'timeNightAwake',
       'leanBodyMass', 'dietaryFiber', 'respiratoryRate',
       'uVExposure',
       'bodyFatPercentage',
       'sleepLatency', 'timeNightAsleep', 'dietaryPhosphorus',
       'dietarySelenium', 'activeEnergyBurned',
       'timeNightRem', 'dietaryFolate',
       'peakExpiratoryFlowRate', 'dietaryVitaminC', 'timeNightInBed',
       'heartRate', 'percentageSleepRem', 'dietaryRiboflavin',
       'distanceWalkingRunning', 'dietarySodium', 'oxygenSaturation',
       'heartRateRecoveryOneMinute', 'dietaryIron', 'timeNightLight', 'sleepWakeups',
       'walkingHeartRateAverage', 'dietaryManganese'
    ],
    'bloodGlucose': [
        'heartRateVariabilitySDNN', 'distanceSwimming',
        'dietaryVitaminE', 'dietarySugar',
       'appleStandTime', 'dietaryFatPolyunsaturated', 'bloodPressureDiastolic',
       'restingHeartRate', 'timeNightDeep', 'bodyMass',
        'dietaryVitaminB12',
       'sleepEfficiency', 'timeInDaylight', 'bloodPressureSystolic',
       'dietaryWater', 'stepCount', 'dietaryPotassium', 'bloodGlucose',
       'dietaryCaffeine', 'dietaryVitaminD', 'dietaryNiacin',
       'appleExerciseTime', 'dietaryCarbohydrates', 'dietaryFatSaturated',
       'dietaryVitaminK', 'wakeupsOver15Minutes',
       'dietaryThiamin', 'mindfulMinutes',
        'environmentalAudioExposure', 'dietaryCholesterol',
       'dietaryCopper', 'dietaryVitaminB6', 'forcedExpiratoryVolume1',
       'dietaryVitaminA', 'bodyMassIndex', 'dietaryEnergyConsumed', 'vO2Max',
       'dietaryZinc', 'headphoneAudioExposure', 'dietaryMagnesium', 'dietaryFatTotal', 'timeSleepAwake',
       'dietaryFatMonounsaturated', 'dietaryPantothenicAcid', 'dietaryProtein',
       'dietaryCalcium', 'timeNightAwake',
       'leanBodyMass', 'dietaryFiber', 'respiratoryRate',
       'uVExposure',
       'bodyFatPercentage',
       'sleepLatency', 'timeNightAsleep', 'dietaryPhosphorus',
       'dietarySelenium', 'activeEnergyBurned',
       'timeNightRem', 'dietaryFolate',
       'peakExpiratoryFlowRate', 'dietaryVitaminC', 'timeNightInBed',
       'heartRate', 'percentageSleepRem', 'dietaryRiboflavin',
       'distanceWalkingRunning', 'dietarySodium', 'oxygenSaturation',
       'heartRateRecoveryOneMinute', 'dietaryIron', 'timeNightLight', 'sleepWakeups',
       'walkingHeartRateAverage', 'dietaryManganese'
    ],
    'restingHeartRate': [
        'heartRateVariabilitySDNN', 'distanceSwimming',
        'dietaryVitaminE', 'dietarySugar',
       'appleStandTime', 'dietaryFatPolyunsaturated', 'bloodPressureDiastolic',
       'restingHeartRate', 'timeNightDeep', 'bodyMass',
        'dietaryVitaminB12',
       'sleepEfficiency', 'timeInDaylight', 'bloodPressureSystolic',
       'dietaryWater', 'stepCount', 'dietaryPotassium', 'bloodGlucose',
       'dietaryCaffeine', 'dietaryVitaminD', 'dietaryNiacin',
       'appleExerciseTime', 'dietaryCarbohydrates', 'dietaryFatSaturated',
       'dietaryVitaminK', 'wakeupsOver15Minutes',
       'dietaryThiamin', 'mindfulMinutes',
       'dietaryCholesterol',
       'dietaryCopper', 'dietaryVitaminB6', 'forcedExpiratoryVolume1',
       'dietaryVitaminA', 'bodyMassIndex', 'dietaryEnergyConsumed', 'vO2Max',
       'dietaryZinc', 'headphoneAudioExposure', 'dietaryMagnesium', 'dietaryFatTotal', 'timeSleepAwake',
       'dietaryFatMonounsaturated', 'dietaryPantothenicAcid', 'dietaryProtein',
        'dietaryCalcium', 'timeNightAwake',
       'leanBodyMass', 'dietaryFiber', 'respiratoryRate',
       'uVExposure',
       'bodyFatPercentage',
       'sleepLatency', 'timeNightAsleep', 'dietaryPhosphorus',
       'dietarySelenium', 'activeEnergyBurned',
       'timeNightRem', 'dietaryFolate',
       'peakExpiratoryFlowRate', 'dietaryVitaminC', 'timeNightInBed',
       'heartRate', 'dietaryRiboflavin',
       'distanceWalkingRunning', 'dietarySodium', 'oxygenSaturation',
       'heartRateRecoveryOneMinute', 'dietaryIron', 'timeNightLight', 'sleepWakeups',
       'walkingHeartRateAverage', 'dietaryManganese'
       ],
    'heartRate': [
        'heartRateVariabilitySDNN', 'distanceSwimming',
        'dietaryVitaminE', 'dietarySugar',
       'appleStandTime', 'dietaryFatPolyunsaturated', 'bloodPressureDiastolic',
       'restingHeartRate', 'timeNightDeep', 'bodyMass',
        'dietaryVitaminB12',
       'sleepEfficiency', 'timeInDaylight', 'bloodPressureSystolic',
       'dietaryWater', 'stepCount', 'dietaryPotassium', 'bloodGlucose',
       'dietaryCaffeine', 'dietaryVitaminD', 'dietaryNiacin',
       'appleExerciseTime', 'dietaryCarbohydrates', 'dietaryFatSaturated',
       'dietaryVitaminK', 'wakeupsOver15Minutes',
       'dietaryThiamin', 'mindfulMinutes',
       'dietaryCholesterol',
       'dietaryCopper', 'dietaryVitaminB6', 'forcedExpiratoryVolume1',
       'dietaryVitaminA', 'bodyMassIndex', 'dietaryEnergyConsumed', 'vO2Max',
       'dietaryZinc', 'headphoneAudioExposure', 'dietaryMagnesium', 'dietaryFatTotal', 'timeSleepAwake',
       'dietaryFatMonounsaturated', 'dietaryPantothenicAcid', 'dietaryProtein',
        'dietaryCalcium', 'timeNightAwake',
       'leanBodyMass', 'dietaryFiber', 'respiratoryRate',
       'uVExposure',
       'bodyFatPercentage',
       'sleepLatency', 'timeNightAsleep', 'dietaryPhosphorus',
       'dietarySelenium', 'activeEnergyBurned',
       'timeNightRem', 'dietaryFolate',
       'peakExpiratoryFlowRate', 'dietaryVitaminC', 'timeNightInBed',
       'heartRate', 'dietaryRiboflavin',
       'distanceWalkingRunning', 'dietarySodium', 'oxygenSaturation',
       'heartRateRecoveryOneMinute', 'dietaryIron', 'timeNightLight', 'sleepWakeups',
       'walkingHeartRateAverage', 'dietaryManganese'
    ],
    'heartRateRecoveryOneMinute': [],
    'heartRateVariabilityRMSSD': [],
    'sleepEfficiency': [
        'heartRateVariabilitySDNN', 'distanceSwimming',
        'dietaryVitaminE', 'dietarySugar',
       'appleStandTime', 'dietaryFatPolyunsaturated', 'bloodPressureDiastolic',
       'restingHeartRate', 'timeNightDeep', 'bodyMass',
        'dietaryVitaminB12',
       'sleepEfficiency', 'timeInDaylight', 'bloodPressureSystolic',
       'dietaryWater', 'stepCount', 'dietaryPotassium', 'bloodGlucose',
       'dietaryCaffeine', 'dietaryVitaminD', 'dietaryNiacin',
       'appleExerciseTime', 'dietaryCarbohydrates', 'dietaryFatSaturated',
       'dietaryVitaminK', 'wakeupsOver15Minutes',
       'dietaryThiamin', 'mindfulMinutes',
       'dietaryCholesterol',
       'dietaryCopper', 'dietaryVitaminB6', 'forcedExpiratoryVolume1',
       'dietaryVitaminA', 'bodyMassIndex', 'dietaryEnergyConsumed', 'vO2Max',
       'dietaryZinc', 'headphoneAudioExposure', 'dietaryMagnesium', 'dietaryFatTotal', 'timeSleepAwake',
       'dietaryFatMonounsaturated', 'dietaryPantothenicAcid', 'dietaryProtein',
        'dietaryCalcium', 'timeNightAwake',
       'leanBodyMass', 'dietaryFiber', 'respiratoryRate',
       'uVExposure',
       'bodyFatPercentage',
       'sleepLatency', 'timeNightAsleep', 'dietaryPhosphorus',
       'dietarySelenium', 'activeEnergyBurned',
       'timeNightRem', 'dietaryFolate',
       'peakExpiratoryFlowRate', 'dietaryVitaminC', 'timeNightInBed',
       'heartRate', 'dietaryRiboflavin',
       'distanceWalkingRunning', 'dietarySodium', 'oxygenSaturation',
       'heartRateRecoveryOneMinute', 'dietaryIron', 'timeNightLight', 'sleepWakeups',
       'walkingHeartRateAverage', 'dietaryManganese'
    ],
    'bloodPressureSystolic': [],
    'bloodPressureDiastolic': [],
    'vO2Max': [
        'heartRateVariabilitySDNN', 'distanceSwimming',
        'dietaryVitaminE', 'dietarySugar',
       'appleStandTime', 'dietaryFatPolyunsaturated', 'bloodPressureDiastolic',
       'restingHeartRate', 'timeNightDeep', 'bodyMass',
        'dietaryVitaminB12',
       'sleepEfficiency', 'timeInDaylight', 'bloodPressureSystolic',
       'dietaryWater', 'stepCount', 'dietaryPotassium', 'bloodGlucose',
       'dietaryCaffeine', 'dietaryVitaminD', 'dietaryNiacin',
       'appleExerciseTime', 'dietaryCarbohydrates', 'dietaryFatSaturated',
       'dietaryVitaminK', 'wakeupsOver15Minutes',
       'dietaryThiamin', 'mindfulMinutes',
        'environmentalAudioExposure', 'dietaryCholesterol',
       'dietaryCopper', 'dietaryVitaminB6', 'forcedExpiratoryVolume1',
       'dietaryVitaminA', 'bodyMassIndex', 'dietaryEnergyConsumed', 'vO2Max',
       'dietaryZinc', 'headphoneAudioExposure', 'dietaryMagnesium', 'dietaryFatTotal', 'timeSleepAwake',
       'dietaryFatMonounsaturated', 'dietaryPantothenicAcid', 'dietaryProtein',
        'dietaryCalcium', 'timeNightAwake',
       'leanBodyMass', 'dietaryFiber', 'respiratoryRate',
       'uVExposure',
       'bodyFatPercentage',
       'sleepLatency', 'timeNightAsleep', 'dietaryPhosphorus',
       'dietarySelenium', 'activeEnergyBurned',
       'timeNightRem', 'dietaryFolate',
       'peakExpiratoryFlowRate', 'dietaryVitaminC', 'timeNightInBed',
       'heartRate', 'percentageSleepRem', 'dietaryRiboflavin',
       'distanceWalkingRunning', 'dietarySodium', 'oxygenSaturation',
       'heartRateRecoveryOneMinute', 'dietaryIron', 'timeNightLight', 'sleepWakeups',
       'walkingHeartRateAverage', 'dietaryManganese'
    ],
    'timeSleepAwake': [
        'heartRateVariabilitySDNN', 'distanceSwimming',
        'dietaryVitaminE', 'dietarySugar',
       'appleStandTime', 'dietaryFatPolyunsaturated', 'bloodPressureDiastolic',
       'restingHeartRate', 'timeNightDeep', 'bodyMass',
        'dietaryVitaminB12',
       'sleepEfficiency', 'timeInDaylight', 'bloodPressureSystolic',
       'dietaryWater', 'stepCount', 'dietaryPotassium', 'bloodGlucose',
       'dietaryCaffeine', 'dietaryVitaminD', 'dietaryNiacin',
       'appleExerciseTime', 'dietaryCarbohydrates', 'dietaryFatSaturated',
       'dietaryVitaminK', 'wakeupsOver15Minutes',
       'dietaryThiamin', 'mindfulMinutes',
       'dietaryCholesterol',
       'dietaryCopper', 'dietaryVitaminB6', 'forcedExpiratoryVolume1',
       'dietaryVitaminA', 'bodyMassIndex', 'dietaryEnergyConsumed', 'vO2Max',
       'dietaryZinc', 'headphoneAudioExposure', 'dietaryMagnesium', 'dietaryFatTotal', 'timeSleepAwake',
       'dietaryFatMonounsaturated', 'dietaryPantothenicAcid', 'dietaryProtein',
        'dietaryCalcium', 'timeNightAwake',
       'leanBodyMass', 'dietaryFiber', 'respiratoryRate',
       'uVExposure',
       'bodyFatPercentage',
       'sleepLatency', 'timeNightAsleep', 'dietaryPhosphorus',
       'dietarySelenium', 'activeEnergyBurned',
       'timeNightRem', 'dietaryFolate',
       'peakExpiratoryFlowRate', 'dietaryVitaminC', 'timeNightInBed',
       'heartRate', 'dietaryRiboflavin',
       'distanceWalkingRunning', 'dietarySodium', 'oxygenSaturation',
       'heartRateRecoveryOneMinute', 'dietaryIron', 'timeNightLight', 'sleepWakeups',
       'walkingHeartRateAverage', 'dietaryManganese'
    ],
    'timeNightDeep': [
        'heartRateVariabilitySDNN', 'distanceSwimming',
        'dietaryVitaminE', 'dietarySugar',
       'appleStandTime', 'dietaryFatPolyunsaturated', 'bloodPressureDiastolic',
       'restingHeartRate', 'timeNightDeep', 'bodyMass',
        'dietaryVitaminB12',
       'sleepEfficiency', 'timeInDaylight', 'bloodPressureSystolic',
       'dietaryWater', 'stepCount', 'dietaryPotassium', 'bloodGlucose',
       'dietaryCaffeine', 'dietaryVitaminD', 'dietaryNiacin',
       'appleExerciseTime', 'dietaryCarbohydrates', 'dietaryFatSaturated',
       'dietaryVitaminK', 'wakeupsOver15Minutes',
       'dietaryThiamin', 'mindfulMinutes',
       'dietaryCholesterol',
       'dietaryCopper', 'dietaryVitaminB6', 'forcedExpiratoryVolume1',
       'dietaryVitaminA', 'bodyMassIndex', 'dietaryEnergyConsumed', 'vO2Max',
       'dietaryZinc', 'headphoneAudioExposure', 'dietaryMagnesium', 'dietaryFatTotal', 'timeSleepAwake',
       'dietaryFatMonounsaturated', 'dietaryPantothenicAcid', 'dietaryProtein',
        'dietaryCalcium', 'timeNightAwake',
       'leanBodyMass', 'dietaryFiber', 'respiratoryRate',
       'uVExposure',
       'bodyFatPercentage',
       'sleepLatency', 'timeNightAsleep', 'dietaryPhosphorus',
       'dietarySelenium', 'activeEnergyBurned',
       'timeNightRem', 'dietaryFolate',
       'peakExpiratoryFlowRate', 'dietaryVitaminC', 'timeNightInBed',
       'heartRate', 'dietaryRiboflavin',
       'distanceWalkingRunning', 'dietarySodium', 'oxygenSaturation',
       'heartRateRecoveryOneMinute', 'dietaryIron', 'timeNightLight', 'sleepWakeups',
       'walkingHeartRateAverage', 'dietaryManganese'
    ],
    'timeNightLight': [
        'heartRateVariabilitySDNN', 'distanceSwimming',
        'dietaryVitaminE', 'dietarySugar',
       'appleStandTime', 'dietaryFatPolyunsaturated', 'bloodPressureDiastolic',
       'restingHeartRate', 'timeNightDeep', 'bodyMass',
        'dietaryVitaminB12',
       'sleepEfficiency', 'timeInDaylight', 'bloodPressureSystolic',
       'dietaryWater', 'stepCount', 'dietaryPotassium', 'bloodGlucose',
       'dietaryCaffeine', 'dietaryVitaminD', 'dietaryNiacin',
       'appleExerciseTime', 'dietaryCarbohydrates', 'dietaryFatSaturated',
       'dietaryVitaminK', 'wakeupsOver15Minutes',
       'dietaryThiamin', 'mindfulMinutes',
       'dietaryCholesterol',
       'dietaryCopper', 'dietaryVitaminB6', 'forcedExpiratoryVolume1',
       'dietaryVitaminA', 'bodyMassIndex', 'dietaryEnergyConsumed', 'vO2Max',
       'dietaryZinc', 'dietaryMagnesium', 'dietaryFatTotal', 'timeSleepAwake',
       'dietaryFatMonounsaturated', 'dietaryPantothenicAcid', 'dietaryProtein',
        'dietaryCalcium', 'timeNightAwake',
       'leanBodyMass', 'dietaryFiber', 'respiratoryRate',
       'uVExposure',
       'bodyFatPercentage',
       'sleepLatency', 'timeNightAsleep', 'dietaryPhosphorus',
       'dietarySelenium', 'activeEnergyBurned',
       'timeNightRem', 'dietaryFolate',
       'peakExpiratoryFlowRate', 'dietaryVitaminC', 'timeNightInBed',
       'heartRate', 'dietaryRiboflavin',
       'distanceWalkingRunning', 'dietarySodium', 'oxygenSaturation',
       'heartRateRecoveryOneMinute', 'dietaryIron', 'timeNightLight', 'sleepWakeups',
       'walkingHeartRateAverage', 'dietaryManganese'
    ],
    'timeNightRem': [
        'heartRateVariabilitySDNN', 'distanceSwimming',
        'dietaryVitaminE', 'dietarySugar',
       'appleStandTime', 'dietaryFatPolyunsaturated', 'bloodPressureDiastolic',
       'restingHeartRate', 'timeNightDeep', 'bodyMass',
        'dietaryVitaminB12',
       'sleepEfficiency', 'timeInDaylight', 'bloodPressureSystolic',
       'dietaryWater', 'stepCount', 'dietaryPotassium', 'bloodGlucose',
       'dietaryCaffeine', 'dietaryVitaminD', 'dietaryNiacin',
       'appleExerciseTime', 'dietaryCarbohydrates', 'dietaryFatSaturated',
       'dietaryVitaminK', 'wakeupsOver15Minutes',
       'dietaryThiamin', 'mindfulMinutes',
       'dietaryCholesterol',
       'dietaryCopper', 'dietaryVitaminB6', 'forcedExpiratoryVolume1',
       'dietaryVitaminA', 'bodyMassIndex', 'dietaryEnergyConsumed', 'vO2Max',
       'dietaryZinc', 'dietaryMagnesium', 'dietaryFatTotal', 'timeSleepAwake',
       'dietaryFatMonounsaturated', 'dietaryPantothenicAcid', 'dietaryProtein',
        'dietaryCalcium', 'timeNightAwake',
       'leanBodyMass', 'dietaryFiber', 'respiratoryRate',
       'uVExposure',
       'bodyFatPercentage',
       'sleepLatency', 'timeNightAsleep', 'dietaryPhosphorus',
       'dietarySelenium', 'activeEnergyBurned',
       'timeNightRem', 'dietaryFolate',
       'peakExpiratoryFlowRate', 'dietaryVitaminC', 'timeNightInBed',
       'heartRate', 'dietaryRiboflavin',
       'distanceWalkingRunning', 'dietarySodium', 'oxygenSaturation',
       'heartRateRecoveryOneMinute', 'dietaryIron', 'timeNightLight', 'sleepWakeups',
       'walkingHeartRateAverage', 'dietaryManganese'
    ],
    'bodyFatPercentage': [],
    'leanBodyMass': [], 
}