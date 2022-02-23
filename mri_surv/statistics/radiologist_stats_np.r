# load things -----
library(dplyr)
library(tibble)
library(readr)
library(tidyr)
library(reshape2)
library(stringr)
library(caret)


setwd('C://Users/micha/Sync/Research/mri-pet/mri_surv/')
tb <- read_csv('./metadata/data_raw/MCIADSubtypeAssessment_weighted_nosub.csv',
                 col_types = cols( 
                  .default=col_factor(),
                  mesial_temp_avg=col_number(),
                 temporal_lobe_other_avg=col_number(),
                 insula_avg=col_number(),
                 frontal_avg=col_number(),
                 cingulate_avg=col_number(),
                 occipital_avg=col_number(),
                 parietal_avg=col_number(),
                 id=col_factor(),
                 id2=col_factor(),
                 rev_initials=col_factor()
               ))

decoder <- read_csv('./metadata/data_processed/shuffled_mri_names.csv')

# reorganize data w/in table -------------------------------------

tb <- tb %>%
  rowwise() %>%
  mutate(Subtype=(decoder[['Cluster Idx']][decoder[['...1']] == id]))

tb$Subtype <- as.factor(recode(tb$Subtype, '0'='H','1'='IH','2'='IL','3'='L'))

tb <- tb %>%
  mutate(Reviewer=rev_initials) %>%
  dplyr::select(-c(rev_initials))
tb$Reviewer <- recode(tb$Reviewer, 'ABP'='1', 'AZM'='2', 'JES'='3','MJS'='4','PHL'='5')
tb <- tb %>%
  dplyr::select(-c(id2, ...1))
tb <- tb %>%
  pivot_longer(cols=-c(Reviewer,Subtype,id), names_to = 'Region', values_to = 'Grade')
tb$Region <- as.factor(
  recode(tb$Region, 'cingulate_avg'='Cingulate','frontal_avg'='Frontal',
                    'insula_avg'='Insula','mesial_temp_avg'='Mesial Temporal',
                    'temporal_lobe_other_avg'='Temporal (other)', 'occipital_avg'= 'Occipital',
                    'parietal_avg'='Parietal')
)

# statistics -----------------------------------------------------------------------

# Take the average grade for each entire lobe within each cluster, plot it

library(ggplot2)
library(svglite)
#trying to loop over each lobe, manually change the reviewer


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

# Rank sum tests, each reviewer's scoring by lobe for each cluster-------

tb2 <- tb

tb2$Subtype <- as.factor(recode(tb$Subtype, 'H'='H','IH'='H','IL'='L','L'='L'))

p <- c()
stat <- c()
est <- c()
lobe <- c()

for (region in unique(tb$Region)) {
  sub.sub.tbl <- tb2 %>%
    filter(Region == region) %>%
    group_by(Subtype, id) %>%
    summarise(Grade.Mn=mean(Grade)) %>%
    ungroup()
  st <- wilcox_test(Grade.Mn~Subtype, data=sub.sub.tbl, method=asymptotic())
  p <- c(p, st@distribution@pvalue(st@statistic@teststatistic))
  stat <- c(stat, st@statistic@teststatistic)
  lobe <- c(lobe, region)
}

names(stat) <- NULL
names(est) <- NULL
wil.stats <- tibble(lobe, est, stat, p)

wil.stats$p.correct <- p.adjust(wil.stats$p, method="fdr")

write_tsv(wil.stats, file='./results/radiologist_gmv_comparisons.tsv')

# now do test as types in step ---------------------------------------------

tb2 <- tb

tb2$Subtype <- as.factor(recode(tb$Subtype, 'H'='H','IH'='H','IL'='L','L'='L'))

p <- c()
stat <- c()
est <- c()
lobe <- c()

for (region in unique(tb$Region)) {
  sub.sub.tbl <- tb2 %>%
    filter(Region == region) %>%
    group_by(Subtype, id) %>%
    summarise(Grade.Mn=mean(Grade)) %>%
    ungroup()
  st <- wilcox_test(Grade.Mn~Subtype, data=sub.sub.tbl, method=asymptotic())
  p <- c(p, st@distribution@pvalue(st@statistic@teststatistic))
  stat <- c(stat, st@statistic@teststatistic)
  lobe <- c(lobe, region)
}

names(stat) <- NULL
names(est) <- NULL
wil.stats <- tibble(lobe, est, stat, p)

wil.stats$p.correct <- p.adjust(wil.stats$p, method="fdr")

write_tsv(wil.stats, file='./results/radiologist_gmv_comparisons.tsv')

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

