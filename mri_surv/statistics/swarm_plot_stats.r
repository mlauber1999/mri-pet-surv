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

# ------------------------------------------------------------------
tb <- read_csv('./metadata/data_processed/swarm_plot_data_raw.csv',
               col_types = cols(
                 'Cluster Idx'=col_factor(),
                 Region=col_factor(),
                 RID=col_factor(),
                 Dataset=col_factor()))

tb <- tb %>%
  filter(Dataset == 'NACC') %>%
  dplyr::select('RID','Region','Shap Value', 'Cluster Idx')

tb <- rename(tb, 'Subtype' = 'Cluster Idx')
tb <- rename(tb, 'Shap' = 'Shap Value')


tb$logShap <- transform(tb$Shap)

tb$Cluster <- recode(tb$Subtype, '0'='H','1'='IH','2'='IL','3'='L')

## Shap value ~ Cluster Idx + (1|RID) + (1|Region)]
mdl <- lmer('logShap ~ Subtype + (1|RID)+(1|Region)', data=tb)

results <- lsmeans(mdl, pairwise~Subtype, adjust="sidak")
results.contrasts <- as_tibble(results$contrasts)
results.lsmeans <- as_tibble(results$lsmeans)
write_tsv(results.contrasts, './results/shap_lme_results_contrasts.txt', append=FALSE)
write_tsv(results.lsmeans, './results/shap_lme_results_emmeans.txt', append=FALSE)
sink('./results/shap_lme_results_summary.txt',append=FALSE)
print(anova(mdl))
print(summary(mdl))
sink()

# Plot diagnostics for the above ---------------------------
library(qqplotr)
tb$Cluster <- ordered(tb$Cluster, levels=c('H','IH','IL','L'))

plt1 <- ggplot(data.frame(Residuals=residuals(mdl, type='pearson'), Subtype=tb$Cluster),
               aes(x=Subtype, y=Residuals)) +
  geom_boxplot(notch=TRUE)

plt2 <- ggplot(data.frame(Residuals=residuals(mdl, type='pearson'), Region=tb$Region),
               aes(x=Region, y=Residuals)) +
  geom_boxplot(notch=TRUE)

res <- residuals(mdl,'pearson')
names(res) <- NULL

plt3 <- ggplot(data.frame(Residuals=res), aes(sample=Residuals)) +
  stat_qq_band() +
  stat_qq_line() +
  stat_qq_point()

plt4 <- ggplot(data.frame(Fitted=fitted(mdl),Residuals=res),
               aes(x=Fitted,y=Residuals)) +
  geom_point() +
  theme_bw()

plt <- grid.arrange(plt1, plt2, plt3, plt4, nrow=2, ncol=2)

ggsave('./figures/lme_diagnostics1.svg',plot=plt, device='svg', dpi='retina')