def categorize_columns(data):
    data["season"] = data.season.astype("category")
    data["mnth"] = data.mnth.astype("category")
    data["hr"] = data.hr.astype("category")
    data["weekday"] = data.weekday.astype("category")
    data["weathersit"] = data.weathersit.astype("category")
    data["workingday"] = data.workingday.astype("int8")
    data["holiday"] = data.holiday.astype("int8")
    data["yr"] = data.yr.astype("int8")

    return data

def drop_registered(data):
    """
    Drops the registered column from the data frame

    We cannot use this variable for predicting count since it is information that we would not have when using the
    model in production, so we must drop it

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column removed
    """
    print("\tDropping the registered variable since we won't have this information")
    return data.drop("registered", axis=1)


def drop_casual(data):
    """
    Drops the casual column from the data frame

    We cannot use this variable for predicting count since it is information that we would not have when using the
    model in production, so we must drop it

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column removed
    """
    print("\tDropping the causual variable since we won't have this information")
    return data.drop("casual", axis=1)


def drop_date(data):
    """
    Drops the date column from the data frame

    This column contains exclusively redundant information, so we can drop it

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column removed
    """
    print("\tDropping the date variable since this information is encoded in other variables")
    return data.drop("dteday", axis=1)


def drop_instant(data):
    """
    Drops the instant column from the data frame

    This column is essentially just a row number, so we don't need it as a feature

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column removed
    """
    print("\tDropping index variable")
    return data.drop("instant", axis=1)


def year_as_bool(data):
    """
    Converts the year column to a boolean rather than an integer

    This is just a prototype that will be used exclusively on the last quarter of 2012, so we may keep it as a boolean.
    In the future though, this should be a category straight out.

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column updated to the new type
    """
    data = data.copy()
    print("\tConverting year to a boolean variable...")
    data["yr"] = data["yr"].astype("bool")
    return data


def season_as_category(data):
    """
    Converts the season column to a category rather than an integer

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column updated to the new type
    """
    data = data.copy()
    print("\tConverting season to a categorical variable...")
    data["season"] = data["season"].astype("category")
    return data


def month_as_category(data):
    """
    Converts the month column to a category rather than an integer

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column updated to the new type
    """
    data = data.copy()
    print("\tConverting month to a categorical variable...")
    data["mnth"] = data["mnth"].astype("category")
    return data


def weekday_as_category(data):
    """
    Converts the weekday column to a category rather than an integer

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column updated to the new type
    """
    data = data.copy()
    print("\tConverting day of week to a categorical variable...")
    data["weekday"] = data["weekday"].astype("category")
    return data


def hour_as_category(data):
    """
    Converts the hour column to a category rather than an integer

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column updated to the new type
    """
    data = data.copy()
    print("\tConverting hour of day to a categorical variable...")
    data["hr"] = data["hr"].astype("category")
    return data


def holiday_as_bool(data):
    """
    Converts the holiday column to a boolean rather than an integer

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column updated to the new type
    """
    data = data.copy()
    print("\tConverting holiday or not to a boolean variable...")
    data["holiday"] = data["holiday"].astype("bool")
    return data


def working_day_as_bool(data):
    """
    Converts the workingday column to a category rather than an integer

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column updated to the new type
    """
    data = data.copy()
    print("\tConverting holiday or not to a boolean variable...")
    data["workingday"] = data["workingday"].astype("bool")
    return data


def weather_sit_as_category(data):
    """
    Converts the weathersit column to a category rather than an integer

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the column updated to the new type
    """
    data = data.copy()
    print("\tConverting weather situation to a categorical variable...")
    data["weathersit"] = data["weathersit"].astype("category")
    return data
