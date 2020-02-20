rm(list = ls())
for(y in 1:3){
  start_time <- Sys.time()
  library(gurobi)
  
  # This script optimizes trading behavior for a FLEX B.
  # Run per year, adapt flexibility characteristics.
  
  model <-list()
  m <- matrix(rep(1,96), nrow=1, ncol=96, byrow=T)
  m_tri <- lower.tri(matrix(rep(1,9216), nrow=96, ncol=96, byrow=T), diag = TRUE)*matrix(rep(1,9216), nrow=96, ncol=96, byrow=T)
  output <- array(0,dim = c(12,31))
  colnames(output)<- c(1:31)
  rownames(output)<- c(1:12)
  #y=1 #2015=1, 2016=2, 2017=3
  
  #market 
  cost_fin <- 0 
  cost_init <- 0 
  vstd <- 96 
  price_buy <- array(0,dim = c(1,96)) 
  price_latest <- array(0, dim=c(1,96)) 
  v_latest <- array(100, dim=c(1,96))
  v_0 <- array(0,dim=c(1,96)) 
  
  #flexibility option
  W_max <- 1.4
  P_max <- 2.5
  
  #read auction data
  setwd(".../Auktion")
  files_p <- list.files(pattern="prices")
  files_v <- list.files(pattern="volumes")
  auction_p <- read.csv(files_p[y])
  auction_v <- read.csv(files_v[y])
  auction_p[,102:109] <- NULL
  auction_p[,14:17] <- NULL
  auction_v[,102] <- NULL
  auction_v[,14:17] <- NULL
  auction_p[is.na(auction_p)] <- 100000000
  auction_v[is.na(auction_v)] <- 0
  auction_p$Date <-strptime(as.character(auction_p$Delivery.day), "%d/%m/%Y")
  auction_v$Date <-strptime(as.character(auction_p$Delivery.day), "%d/%m/%Y")
  
  #read trading data
  wd =  paste(".../", (y+2014), sep = "")
  setwd(wd)
  files_csv <- list.files(pattern="export_2.csv")
  
  for (k in 1:12) { #12 #month
  
      data <- read.csv(files_csv[k])
      data$X <- NULL
      data <- subset(data,data$product=="VSTD")
  #    data <- subset(data,data$VSTD.from>9*4)
  #    data <- subset(data,data$VSTD.from<=17*4)
      dates <- aggregate(data.frame(count = data$Date), list(value = data$Date), length)
  
      for (j in 1:length(dates$value)) { #days
        print(as.character(dates$value[j]))
        trade_per_day <- subset(data, data$Date == dates$value[j])
        trade_per_day$Time_buy[is.na(trade_per_day$Time_buy)] <- 1
        
        #(re)set data
        revenue=0
        cost_init <- 0 
        cost_fin <- 0
        price_buy <- array(0,dim = c(1,96))
        v_0 <- array(0,dim=c(1,96))
        price_latest <- as.matrix(auction_p[which(auction_p$Date == as.character(dates$value[j])),2:97])
        v_latest <- as.matrix(auction_v[which(auction_v$Date == as.character(dates$value[j])),2:97])
        
        for (i in 0:length(trade_per_day$Date)){ #Transaktionen, i = 0 Initialisierung aus ID Auktion
          price_latest[trade_per_day$VSTD.from[i]] <- trade_per_day$Price..EUR.[i]    
          v_latest[trade_per_day$VSTD.from[i]] <- trade_per_day$Volume..MW.[i]      
          
          vstd <- ifelse(trade_per_day$Time_buy[i] < 536,96,96-floor((trade_per_day$Time_buy[i]-521)/15)) #noch zu handelnde VSTD
          if (i == 0) { vstd = 96 }
          
          v_traded <- ifelse(96-vstd == 0, 0,sum(v_0[1:(96-vstd)]))
            
          #optimize
          model$A <- as.matrix(rbind(m_tri,m_tri)[,(96-vstd+1):96]) #matrix(rep(1,vstd), nrow=1, ncol=vstd, byrow=T)  
          model$obj<- price_latest[(96-vstd+1):96]/4
          model$modelsense <- 'min'
          model$objcon <- sum(v_0[(96-vstd+1):96]*price_latest[(96-vstd+1):96]/4*-1)
          model$rhs <-rep(c(0,((W_max*4)-v_traded)),each=96)
          model$ub <- pmin(v_latest, P_max)[(96-vstd+1):96]
          model$lb <- pmax(-1*v_latest, -1*P_max)[(96-vstd+1):96]                                   
          model$sense  <- c(rep(">=",96),rep("<=",95),"=") 
          params <- list(OutputFlag=0)
          result <- gurobi(model, params)
          
          #print(result$objval)
          #print(result$x)
          
          if (i == 0) { cost_init <- result$objval
          }
          else if (!is.null(result$x)) {
            if(result$objval <= 0) {
              # if (result$objval != 0) {
              #   print(result$objval)
              #   print(cost_fin)}
            v_0 [(96-vstd+1):96]<- result$x 
             #if (i==1){print(v_0)}
            cost_fin <- cost_fin + result$objval 
            for (l in 1:96) { price_buy [1,l] <- ifelse(result$x[l] >0, price_latest[l],price_buy[1,l])}
            }
          }  
          
        }
        
        revenue <- ((cost_fin - cost_init)) * -1
        output[k,j]<-revenue
      }
  }
  
  setwd("...")
  write.csv(output,paste0("bucket_",y,"_",W_max,"_",P_max,".csv")) #2015=1, 2016=2, 2017=3        
  end_time <- Sys.time()
  runtime_total <- end_time - start_time
  runtime_total
}
