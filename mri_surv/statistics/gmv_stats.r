# init ----------------------------
library(dplyr)
library(tibble)
library(readr)
library(tidyr)
library(lme4)
library(lmerTest)
library(broom)
library(emmeans)
library(ggplot2)
library(gridExtra)
library(sjPlot)

setwd('C://Users/micha/Sync/Research/mri-pet/mri_surv/')
transform <- function(x) {
  x = x-min(x)+1
  return(log(x))
}

# model = GMV ~ region*cluster + (cluster|RID) + (region|RID) + (cluster|cohort) + (region|cohort) -------

tb2 <- read_csv('./metadata/data_processed/shap_with_parcellations_long.csv',
                col_types = cols('Cluster Idx'=col_factor(),
                                 Region=col_factor(), RID=col_factor(),
                                 Dataset=col_factor()))
tb2 <- rename(tb2, 'Subtype'='Cluster Idx', 'GMV'='Gray Matter Vol')
tb2$Subtype <- dplyr::recode(tb2$Subtype, '0'='H','1'='IH','2'='IL','3'='L')
tb2 <- tb2 %>%
  filter(Dataset=='ADNI')
tb2 <- tb2 %>%
  mutate(logGMV=transform(GMV))
mdl2 <- lmer('logGMV ~ Region*Subtype + (1|RID)', tb2)

tb2$Subtype <- ordered(tb2$Subtype, levels=c('H','IH','IL','L'))

results <- lsmeans(mdl2, pairwise~Region|Subtype, adjust="sidak")

summ <- data.frame(results$lsmeans)

parsed <- summ %>%
  group_by(Subtype) %>%
  mutate(rank=rank(lsmean), invrank=rank(-lsmean)) %>%
  filter(rank <= 5 | invrank <= 5) %>%
  arrange(rank, .by_group = TRUE)

parsed.tsv <- as_tibble(parsed)
write_tsv(parsed.tsv, './results/lme_gmv_region_by_subtype_mean.txt')

results.bycluster <- lsmeans(mdl2, pairwise~Region|Subtype, adjust="sidak")
results.bycluster <- as_tibble(results.bycluster$contrasts)
write_tsv(results.bycluster, './results/lme_gmv_region_by_subtype_contrast.txt', append=FALSE)

sink('./results/lme_gmv_summary.txt')
print(anova(mdl2))
print(summary(mdl2))
sink()

salient.regions <- unique(parsed$Region)
results <- lsmeans(mdl2, pairwise~Subtype|Region, at=list(Region=salient.regions), adjust="sidak")
lmean.mean <- as_tibble(results$lsmeans)
lmean.contrast <- as_tibble(results$contrasts)
write_tsv(lmean.mean, './results/lme_gmv_subtype_by_region_mean.txt', append=FALSE)
write_tsv(lmean.contrast, './results/lme_gmv_subtype_by_region_contrast.txt', append=FALSE)

# plot model diagnostics ---------------
library(qqplotr)

res <- residuals(mdl2,'pearson')

plt1 <- ggplot(data.frame(Residuals=res, Subtype=tb2$Subtype),
               aes(x=Subtype, y=Residuals)) +
  geom_boxplot(notch=TRUE)

plt2 <- ggplot(data.frame(Residuals=res, Region=tb2$Region),
               aes(x=Region, y=Residuals)) +
  geom_boxplot(notch=TRUE) + 
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

names(res) <- NULL

plt3 <- ggplot(data.frame(Residuals=(res)), aes(sample=Residuals)) +
  stat_qq_band() +
  stat_qq_line() +
  stat_qq_point()

plt4 <- ggplot(data.frame(Fitted=fitted(mdl2),Residuals=res),
               aes(x=Fitted,y=Residuals)) +
  geom_point() +
  theme_bw()

p <- grid.arrange(plt1, plt2, plt3, plt4, nrow=2, ncol=2)

ggsave('./figures/lme_diagnostics2.svg',plot=p, device='svg', dpi='retina')
