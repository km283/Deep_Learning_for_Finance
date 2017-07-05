import csv
from yahoo_finance import Share

company_name= "YHOO"
csv_name = "{}.csv".format(company_name)



def main():
    some_company = Share(company_name)
    some_companies_data = some_company.get_historical('2011-07-04', '2017-07-04')

    with open(csv_name, "w") as f:
        w = csv.DictWriter(f, some_companies[0].keys())
        w.writeheader()
        for index, data in enumerate(some_companies_data):
            w.write_row(data)
    print(len(some_companies_data))





if __name__ == "__main__":
    main()
