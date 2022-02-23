# load things -----
library(dplyr)
library(tibble)
library(readr)

library(dplyr)
library(tibble)
library(readr)

library(lme4)
library(nlme)
library(tidyr)
library(reshape2)
library(stringr)
library(lmerTest)
library(emmeans)
library(ordinal)
library(caret)



setwd('C://Users/micha/Sync/Research/mri-pet/mri_surv/')
tb <- read_csv('./metadata/data_raw/MCIADSubtypeAssessment_weighted_nosub.csv')
decoder <- read_csv('./metadata/data_processed/shuffled_mri_names.csv')

# tests ------------------------

tb2 <- tb %>%
  rowwise() %>%
  mutate(m = (id == id2))

all(tb2[['m']])

# test length of studies are same
l <- tb %>%
  group_by(rev_initials) %>%
  summarise(l = length(rev_initials))

length(unique(l[['l']])) == 1

#test that all are unique

l.unique <- tb %>%
  group_by(rev_initials) %>% summarise(u = length(unique(id)))

all(l.unique[['l']] == 48)

lobes <- c('mesial_temp_avg', 'temporal_lobe_other_avg','insula_avg', 'frontal_avg', 'cingulate_avg','occipital_avg','parietal_avg')

tb <- tb %>%
  rowwise() %>%
  mutate(Cluster=(decoder[['Cluster Idx']][decoder[['...1']] == id]))

# statistics -----------------------------------------------------------------------

# Take the average grade for each entire lobe within each cluster, plot it

library(ggplot2)
library(svglite)
#trying to loop over each lobe, manually change the reviewer

tb$Subtype <- recode(tb$Cluster, '0'='H','1'='IH','2'='IL','3'='L')

tb <- tb %>%
  mutate(Reviewer=rev_initials) %>%
  select(-c(rev_initials))
tb$Reviewer <- recode(tb$Reviewer, 'ABP'='1', 'AZM'='2', 'JES'='3','MJS'='4','PHL'='5')
tb <- tb %>%
  select(-c(id, id2, Cluster, ...1))
tb <- tb %>%
pivot_longer(cols=-c(Reviewer,Subtype), names_to = 'Region', values_to = 'Grade')
tb$Region <- recode(tb$Region, 'cingulate_avg'='Cingulate','frontal_avg'='Frontal',
                    'insula_avg'='Insula','mesial_temp_avg'='Mesial Temporal',
                    'temporal_lobe_other_avg'='Temporal (other)', 'occipital_avg'= 'Occipital',
                    'parietal_avg'='Parietal')

  ggplot(tb) + aes_string(x="Subtype",y="Grade", fill="Reviewer")+
    geom_jitter(height=0, width=0.2, aes(colour=Reviewer, shape=Reviewer))+
    coord_cartesian(ylim = c(0, 3.0))+
    stat_summary(fun=mean, geom='line', aes(group=1),size=2) +
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour='black')
      ) +  # help from https://felixfan.github.io/ggplot2-remove-grid-background-margin/
    theme(axis.text.x = element_text(
      colour = 'black', size = 15),
      axis.title.x=element_text(
        colour='black', size=20
      )) +
    theme(axis.text.y = element_text(
      colour = 'black', size = 15),
      axis.title.y=element_text(
        colour='black', size=20
      )) +
    theme(strip.text = element_text(
      colour = 'black', size = 15)) +
    theme(legend.title = element_text(colour='black',size=15), 
          legend.position=c(0.7, 0.1),
          legend.direction="horizontal",
          legend.text = element_text(colour='black', size=12) )+
    facet_wrap(~Region)
    
    
  ggsave(file=paste0("figures/reviewer_grades.svg"), width=8, height=10)

# Mixed-effects modeling, each reviewer's scoring by lobe for each cluster-------

tb <- read_csv('./metadata/data_raw/MCIADSubtypeAssessment_handed.csv',
                      col_types = cols_only(
                        mesial_temp_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        mesial_temp_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        temporal_lobe_other_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        temporal_lobe_other_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        insula_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        insula_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        frontal_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        frontal_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        cingulate_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        cingulate_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        occipital_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        occipital_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        parietal_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        parietal_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                        id=col_factor(),
                        rev_initials=col_factor(),
                      ))

l_or_r <- function(x) {
  if (!is.na(str_extract(x,regex(pattern='_r_sum')))) {
    return('right')
  } else if (!is.na(str_extract(x,regex(pattern='_l_sum')))) {
    return('left')
  } else {
    return(NA)
  }
}

drop_suffix <- function(x) {
  return(str_remove(x,regex('_[lr]{1}_sum')))
}

tb <- tb %>%
  rowwise() %>%
  mutate(Cluster=(decoder[['Cluster Idx']][decoder[['...1']] == id]))

tb$review <- as.factor(1:dim(tb)[1])

g <- tb %>% 
  melt(id=c('id','rev_initials','Cluster','review'), value.name='Grade')

g <- g %>%
  mutate(hemi=sapply(variable, l_or_r), region=sapply(variable, drop_suffix)) %>%
  select(-variable)

g$Cluster <- as.factor(g$Cluster)
g$rev_initials <- as.factor(g$rev_initials)
g$id <- as.factor(g$id)
g$review <- as.factor(g$review)
g$region <- as.factor(g$region)
g$hemi <- as.factor(g$hemi)
g$Grade <- as.numeric(g$Grade)


# mdl <- lmer(data=g, Grade~1+region*Cluster+(1|id:hemi)+(1|rev_initials))
g$Grade <- as.ordered(g$Grade)

mdl <- clmm(data=g, Grade~1+region*Cluster+(1|id:hemi)+(1|rev_initials))

mdl2 <- clmm(Grade~1+region+Cluster+(1|id:hemi)+(1|rev_initials), data=g)

st.interact <- anova(mdl,mdl2)

sink('./results/radiology_statistics_summary.txt')

cat('Summary\n')

print(summary(mdl))
print(anova(mdl,mdl2))
sink()

lmeans <- lsmeans(mdl, pairwise~Cluster|region, adjust="sidak")

j <-  as_tibble(lmeans$lsmeans)

write_tsv(j, './results/radiology_statistics_emmeans.txt')

k <-  as_tibble(lmeans$contrasts)

write_tsv(k,'./results/radiology_statistics_contrasts.txt')


lmeans <- lsmeans(mdl, pairwise~region|Cluster, adjust="sidak")

j <-  as_tibble(lmeans$lsmeans)

write_tsv(j, './results/radiology_statistics_region_by_cluster_emmeans.tsv')

k <-  as_tibble(lmeans$contrasts)

write_tsv(k,'./results/radiology_statistics_region_by_cluster_contrasts.tsv')


# plotting diagnostics --------------------------
library(qqplotr)
library(gridExtra)
res <- residuals(mdl,'pearson')

plt1 <- ggplot(data.frame(Residuals=res, Subtype=g$Cluster),
               aes(x=Subtype, y=Residuals)) +
  geom_boxplot(notch=TRUE)

plt2 <- ggplot(data.frame(Residuals=res, Region=g$region),
               aes(x=Region, y=Residuals)) +
  geom_boxplot(notch=TRUE) + 
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

names(res) <- NULL

plt3 <- ggplot(data.frame(Residuals=(res)), aes(sample=Residuals)) +
  stat_qq_band() +
  stat_qq_line() +
  stat_qq_point()

plt4 <- ggplot(data.frame(Fitted=fitted(mdl),Residuals=res),
               aes(x=Fitted,y=Residuals)) +
  geom_point() +
  theme_bw()

p <- grid.arrange(plt1, plt2, plt3, plt4, nrow=2, ncol=2)

ggsave('./figures/lme_diagnostics4.svg',plot=p, device='svg', dpi='retina')


# icc's D ---------------------------------------------------------------
library("vcd")
library("irr")
# load table with individual ratings

tb.handed <- read_csv('./metadata/data_raw/MCIADSubtypeAssessment_handed.csv',
               col_types = cols_only(
                 mesial_temp_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 mesial_temp_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 temporal_lobe_other_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 temporal_lobe_other_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 insula_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 insula_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 frontal_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 frontal_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 cingulate_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 cingulate_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 occipital_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 occipital_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 parietal_l_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 parietal_r_sum=col_factor(ordered = TRUE, levels = c('0.0','1.0','2.0','3.0')),
                 id=col_factor(),
                 rev_initials=col_factor()
               ))
decoder <- read_csv('./metadata/data_processed/shuffled_mri_names.csv',
                    col_types=cols_only(
                      '...1'=col_factor(),
                      'Cluster Idx'=col_factor()
                    ))
tb.handed <- tb.handed %>%
  rowwise() %>%
  mutate(Cluster=(decoder[['Cluster Idx']][decoder[['...1']] == id]))

brain.regions = c('mesial_temp_l_sum',
                  'mesial_temp_r_sum',
                  'temporal_lobe_other_l_sum',
                  'temporal_lobe_other_r_sum',
                  'insula_l_sum',
                  'insula_r_sum',
                  'frontal_l_sum',
                  'frontal_r_sum',
                  'cingulate_l_sum',
                  'cingulate_r_sum',
                  'occipital_l_sum',
                  'occipital_r_sum',
                  'parietal_l_sum',
                  'parietal_r_sum')

# iterate through each column
# make tibble 



icc.list <- vector("list", length(brain.regions)+1)
for (i in 1:length(brain.regions)) {
  reg <- brain.regions[[i]]
  tb.current <- tb.handed[c(reg, 'rev_initials', 'id')]
  tb.current[[reg]] <- as.numeric(tb.current[[reg]])
  tb.current <- tb.current %>%
    spread(rev_initials, reg) %>%
    select(-id)
  icc.coef <- icc(tb.current, model='twoway', type='agreement', unit='single')
  nm.list <- c(unlist(icc.coef))
  nm.list$region <- reg
  icc.list[[i]] <- nm.list
}

tb.current <- tb.handed %>%
  melt(id=c('rev_initials', 'id')) %>%
  mutate(value=as.numeric(value)) %>%
  spread(rev_initials, value) %>%
  select(-c(variable,id))
icc.coef <- icc(tb.current, model='twoway', type='agreement', unit='single')
nm.list <- unlist(icc.coef)
nm.list$region <- "All"
icc.list[[length(brain.regions)+1]] <- nm.list

library(data.frame)
g <- rbindlist(icc.list)
g <- as_tibble(g)
write_csv(g, 'results/interrater_reliability_results.csv')

