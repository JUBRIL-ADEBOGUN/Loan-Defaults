After loading the dataset.
datasets hgas no null values and contain 2 datetime columns, 
## Data validation
Total_Amount,Total_Amount_to_Repay, Amount_Funded_By_Lender,Lender_portion_to_be_repaid are converted to integer and duration into weeks.
customer_id,tbl_loan_id	lender_id, loan type, and New_versus_Repeat are convert to category.
## Feature creations.
interest: total_Amount and Total_Amount_to_repay
lender_interest: Lender_portion_to_be_repaid and Amount_Funded_By_Lender
year: the year the loan was disbursed.
official_interest: official interest rate from the economic indicators data
interest_status: {overcharge:if lender_interest > loan_interest, under_charge: lender_interest<loan_interest, normal: lender_interest == loan_id}
*num_loan: cummulative sum of number of each loan type by a customer.*

## Fearures preprocessing.
use ordinsl encoding for columns with string entry
