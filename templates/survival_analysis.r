library(survival)
library(ranger)
men = read.table('men.csv')
women=read.table('women.csv')
divorces=read.table('divorce.csv')
# Kaplan meier fit for married in the 35-44 year olds
km_3544_fit <- survfit(Surv(year,all_3544),data=divorces)
#summary(km_fit, times = c(1900 + 10*(6:10)))
autoplot(km_3544_fit)

# Kaplan meier fit for married in the middle class 45-54 year olds
km_4554_fit <- survfit(Surv(year,mid_4554),data=divorces)
autoplot(km_4554_fit)

# Kaplan meier fit for married in the middle class 45-54 year olds
km_poor_3544_fit <- survfit(Surv(year,poor_3544),data=divorces)
autoplot(km_poor_3544_fit)
