-- Create Database
CREATE DATABASE shoppers_db;
USE shoppers_db;

-- Create Table
CREATE TABLE online_shoppers (
    Administrative INT,
    Administrative_Duration FLOAT,
    Informational INT,
    Informational_Duration FLOAT,
    ProductRelated INT,
    ProductRelated_Duration FLOAT,
    BounceRates FLOAT,
    ExitRates FLOAT,
    PageValues FLOAT,
    SpecialDay FLOAT,
    Month VARCHAR(10),
    OperatingSystems INT,
    Browser INT,
    Region INT,
    TrafficType INT,
    VisitorType VARCHAR(20),
    Weekend BOOLEAN,
    Revenue BOOLEAN
);

SELECT * FROM online_shoppers;

SELECT Revenue, COUNT(*) FROM online_shoppers GROUP BY Revenue;

COMMIT;
