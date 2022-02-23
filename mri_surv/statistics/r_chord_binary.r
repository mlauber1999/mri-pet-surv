library(reshape)
library(viridis)
library(circlize)
library(dplyr)
library(tibble)
library(svglite)

palette <- viridis_pal(alpha=1, begin=0, end=1, direction=1, option='H')
color.ramp <- colorRamp2(seq(-0.5,0.5,0.05), colors=palette(21), space='HSV', transparency=0)
color.ramp.invisible <- colorRamp2(seq(-0.5, 0.5, 0.05), colors=palette(21), space='HSV', transparency = 1)
color.list <- palette(21)
names(color.list) <- paste0(seq(-0.5, 0.5, 0.05))

sig.link <- function(tbl, top.region) {
  #Rich SCriven response in https://stackoverflow.com/questions/22673335/check-if-each-row-of-a-data-frame-is-contained-in-another-data-frame
  bool.vec <- do.call(paste0, tbl) %in% do.call(paste0, top.region)
  return(bool.vec)
}

top_and_bottom_five <- function(tbl) {
  ord <- order(tbl$PartialCorr)
  tbl <- tbl[ord,]
  top <- head(tbl, 6)
  bottom <- tail(tbl, 6)
  tbl$Significant <- NULL
  rownames(tbl) <- NULL
  sig.top <- sig.link(tbl, top)
  sig.bot <- sig.link(tbl, bottom)
  sig <- sig.top | sig.bot
  tbl$Significant <- ifelse(sig == TRUE, 1, 0)
  tbl <- tbl[order(ord),]
  return(tbl[c('Origin','Regions','Significant')])
}

centrality.all <- tibble(Origin=character(), sum=numeric(), Cluster=numeric())

for (i in c(0,1,2,3)) {
  tbl <- read.csv(paste0(
    './metadata/data_processed/chord_diagram_data_cluster_cluster', i, '.csv', sep=''))
  tbl <- rename(tbl, Origin=Region)
  centrality <- group_by(tbl, Origin) %>% summarise(sum=sum(abs(PartialCorr)))
  centrality.deframed <- deframe(centrality)
  centrality$Cluster <- i
  centrality.all <- bind_rows(centrality.all, centrality)
  tbl.extrema <- top_and_bottom_five(tbl)
  tbl.extrema <- adjacencyList2Matrix(tbl.extrema, square=TRUE)
  tbl <- adjacencyList2Matrix(tbl, square=TRUE)
  col.top <- ifelse(tbl.extrema == 1 & tbl > 0, '#ff0000FF', '#ff00FF00')
  col <- ifelse(tbl.extrema == 1 & tbl < 0, '#0000FFFF', col.top)
  link.border <- ifelse((col == '#0000FFFF')| (col == '#ff0000FF'), "#000000FF", "#00000000")
  grid.col <- rep("green",length(centrality.deframed))
  names(grid.col) <- names(centrality.deframed)
  svglite(paste0("figures/figure5/cord_diagram_cluster",i,".svg",sep=''),width=6, height=6)
  par(cex=1.2)
  circos.par(circle.margin=.4)
  # trick: https://stackoverflow.com/questions/30432224/r-circlize-chord-graph-with-empty-sectors
  c <- chordDiagram(tbl,
                    directional=0,
                    symmetric=TRUE,
                    keep.diagonal=FALSE,
                    col=col,
                    link.border=link.border,
                    grid.col=grid.col,
                    transparency=0.5,
                    annotationTrack=c("grid"),
                    grid.border='black',
  )
  # the below w/ help from: https://github.com/jokergoo/circlize/issues/65
  circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
    x1 <- mean(CELL_META$cell.xlim)
    y1 <- CELL_META$cell.ylim[2]
    t <- circlize(x1, y1)
    t <- reverse.circlize(t[1],t[2])
    circos.text(t, labels=CELL_META$sector.index, facing="downward")
  }, bg.border = NA)
  circos.clear()
}
write.table(centrality.all, file="./results/centralities.csv")