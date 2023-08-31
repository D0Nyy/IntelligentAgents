from tkinter import filedialog as fd
import tkinter as tk
import uuid
import json
import random
import heapq
import time

buildings = ["W","T","A","G","*"]
def astar(start,goal,obstacles, knowledge)->list:
        def heuristic(a, b):
            return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        
        locations = knowledge + [start, goal]
        
        # Initialize data
        open_list = []
        closed_set = set()
        heapq.heappush(open_list, (0, start, []))

        while open_list:
            # Pop the item with the lowest cost
            current_cost, current_node, current_path = heapq.heappop(open_list)

            # Check if the current node is the plan goal
            if current_node == goal:
                return current_path
            
            # Add the current node to the closed set
            closed_set.add(current_node)
            
            # Generate neighbors
            for movement in [(0,1),(1,0),(0,-1),(-1,0)]:
                neighbor = (current_node[0] + movement[0], current_node[1] + movement[1])
                
                # Check if the neighbor is within the city's boundaries
                if neighbor in locations:
                    # Check if the neighbor is an obstacle or in the closed set
                    if neighbor not in obstacles and neighbor not in closed_set:
                        # Calculate the cost to the neighbor
                        new_cost = current_cost + 1
                        # Calculate the total estimated cost
                        total_cost = new_cost + heuristic(neighbor, goal)
                        # Check if the neighbor is already in the open list
                        # and has a lower cost. If so, update its cost and path.
                        updated = False
                        for i, (cost, node, path) in enumerate(open_list):
                            if node == neighbor and cost > total_cost:
                                open_list[i] = (total_cost, neighbor, current_path + [neighbor])
                                updated = True
                                break
                        if not updated:
                            # Add the neighbor to the open list
                            heapq.heappush(open_list, (total_cost, neighbor, current_path + [neighbor]))
                            closed_set.add(neighbor) # Mark the neighbor as visited

        # No path found
        return None

class Agent():
    def __init__(self,city,agentHouse:tuple[str,tuple[int,int]],planFile:str) -> None:
        self.city = city
        self.name = f"IA{agentHouse[0]}_{uuid.uuid4()}"
        self.house = agentHouse
        self.loc = [agentHouse[1][0],agentHouse[1][1]] # get location and make it a list
        self.plan = self.getPlans(planFile)
        self.visited = [tuple(self.loc)] #starting location
        self.knowledge = [tuple(self.loc)] + list(filter(lambda loc: (loc[0] not in [-1,len(self.city.grid[0])]) and (loc[1] not in [-1,len(self.city.grid[0])]),
                                                          [(self.loc[0] + 1, self.loc[1]),(self.loc[0] - 1, self.loc[1]),(self.loc[0], self.loc[1] + 1),
                                                           (self.loc[0], self.loc[1] - 1)])) # initial knowledge (starting + next ones)
        self.selected = False
        self.activeRoute = []
        self.knowledgeTransactions = 0

    def getPlans(self,planFile:str)->list[str]:
        try:
            with open(planFile,"r") as f:
                data = json.load(f)
                for plan in data:
                    if plan["Agent"] == self.name[2]:
                        return plan["Plan"]
        except Exception as e:
            raise Exception("Invalid plan format")
                
    
    def move(self,newLoc):
        self.loc = newLoc
        if tuple(newLoc) not in self.visited:
            self.visited.append(tuple(self.loc))
        self.updateKnowledge([newLoc])

    def updateKnowledge(self,blocks:list):
        for block in blocks:
            if tuple(block) not in self.knowledge:
                self.knowledge.append(tuple(block))
        self.checkKnowledgeForGoal()

    def checkKnowledgeForGoal(self):
        for blockCords in self.knowledge:
            if (len(self.plan) != 0) and (self.city.grid[blockCords[1]][blockCords[0]] == self.plan[0]):
                self.goToLocation(blockCords)
                return

    def goToLocation(self,location):
        obstacles = []
        for blockCords in self.knowledge: #Get known buildings 
            if self.city.grid[blockCords[1]][blockCords[0]] in buildings:
                obstacles.append(blockCords)
                if blockCords == location:
                    location = (blockCords[0],blockCords[1]) # set location next block

        if len(self.plan) !=0:
            obstacles.remove(location)
            self.activeRoute = astar(tuple(self.loc),location,obstacles,self.knowledge)[0:-1]
        else: self.activeRoute = astar(tuple(self.loc),location,obstacles,self.knowledge)

    def lookAround(self, nearbyBlocks:list[str], nearbyAgents):
        # Check if building in plan is visible
        if len(self.plan) != 0:
            for block in nearbyBlocks:
                if (len(self.plan) != 0) and ((block) == self.plan[0]):
                    self.plan.pop(0)
                    if len(self.plan) == 0:
                        self.goHome()
                        break
                    else:
                        self.checkKnowledgeForGoal()
        
        # Trade info with nearby agents
        for agent in nearbyAgents:
            self.tradeInfo(agent)

    def tradeInfo(self,agent):
        self.knowledgeTransactions += 1
        agent.knowledgeTransactions += 1
        #joinedKnowledge = list(set(self.knowledge).union(set(agent.knowledge)))
        self.updateKnowledge(agent.knowledge)
        agent.updateKnowledge(self.knowledge)

    def goHome(self):
        self.goToLocation(self.house[1])

class City():
    def __init__(self,mapRows, planFile:str, agentN:int=None) -> None:
        self.grid = mapRows
        self.size = (len(self.grid),len(self.grid[0]))

        # Check map size
        for row in self.grid:
            if len(row) == self.size[0]:continue
            else: raise Exception("Map size is not consistent")

        self.cityMap = '\n'.join(self.grid)
        self.agents = []
        self.agentHouses = {}
        # get Agent Houses and create agents
        self.getAgentHouses()
        self.createAgents(agentN,planFile)

    def createAgents(self,agentN,planFile:str)->None:
        # Create Agents
        if (agentN == None) or (agentN <= len(self.agentHouses)): # 1 agent per house
            for agentHouse in self.agentHouses.items():
                self.agents.append(Agent(self,agentHouse,planFile))
        elif agentN > len(self.agentHouses): # if agents N > than the houses in the map
            for agentHouse in self.agentHouses.items():
                for _ in range(int(agentN/len(self.agentHouses))):
                    self.agents.append(Agent(self,agentHouse,planFile))
            if(agentN%len(self.agentHouses) != 0): # if mod(N,H) == 0 
                currI = 0
                for _ in range(agentN%len(self.agentHouses)):
                    self.agents.append(Agent(self,(str(currI),self.agentHouses[str(currI)]),planFile))
                    if(currI == len(self.agentHouses)):
                        currI = 0
                        continue
                    currI +=1

    def getAgentHouses(self)-> None:
        self.agentHouses = {}
        for y, row in enumerate(self.grid):
            for x, spot in enumerate(row):
                if spot in [str(i) for i in range(10)]:
                    self.agentHouses[spot] = (x,y)
    
    def getBuildings(self):
        buildingsLocs = []
        for y,row in enumerate(self.grid):
            for x,block in enumerate(row):
                if block in buildings:
                    buildingsLocs.append((x,y))
        return buildingsLocs

class Application(tk.Toplevel):
    def __init__(self, master, mapFile:str, planFile:str, agentsN:int = None, blockSize=40) -> None:
        super().__init__(master = master)
        self.started = False
        self.finished = False
        self.title("Intelligent Agents")
        self.blockSize = blockSize
        self.create_city(mapFile,planFile,agentsN)
        self.geometry(f"{(self.city.size[1] * blockSize) + 300}x{(self.city.size[0] * blockSize)}")
        self.resizable(False,False)

        self.build_interface()
        self.createControls()
        self.printInfo()

    def printInfo(self):
        print(f"Map Size: {self.city.size}")
        print(f"Map Structure:\n{self.city.cityMap}")
        print("Agent Houses:")
        print(self.city.agentHouses)
        print("Agents:")
        for agent in self.city.agents:
            print(agent.name)
        self.beginSimulation()

    def drawPath(self,path):
        # Show knowledge
        block_items = self.canvas.find_withtag("block")
        for blockID in block_items:
            blockLoc = (int(self.canvas.gettags(blockID)[2]),int(self.canvas.gettags(blockID)[1]))
            if blockLoc in path:
                self.canvas.itemconfigure(blockID,fill="green")
        return

    def build_interface(self):
        self.canvasFrame = tk.Frame(self,bg="white")
        self.canvasFrame.grid(row=0,column=0, sticky="nsew")
        self.canvas = tk.Canvas(self.canvasFrame, width=self.city.size[1] * self.blockSize, height=self.city.size[0] * self.blockSize, bg="white")
        self.canvas.pack(anchor="w")

        self.panel_frame = tk.Frame(self, bg="lightgray")
        self.panel_frame.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1,weight=1)

        tk.Label(self.panel_frame,foreground="Black", background="lightgray",text=f"Selected Agent:").pack()
        self.labelSelected = tk.Label(self.panel_frame,foreground="Black", background="lightgray",text=f"None")
        self.labelSelected.pack()

        self.labelPlan = tk.Label(self.panel_frame,foreground="Black", background="lightgray",text=f"Plan: None")
        self.labelPlan.pack()

        self.labelCords = tk.Label(self.panel_frame,foreground="Black", background="lightgray", text=f"BlockLoc:(-,-)")
        self.labelCords.pack(side="bottom",anchor="w")

        self.labelSteps = tk.Label(self.panel_frame,foreground="Black", background="lightgray",text=f"Unique Steps: None")
        self.labelSteps.pack()
        self.labelKnowledgeN = tk.Label(self.panel_frame,foreground="Black", background="lightgray",text=f"Knowledge Exchanges: None")
        self.labelKnowledgeN.pack()

        tk.Label(self.panel_frame,foreground="Black", background="lightgray",text=f"Press 'x' to make a move if paused.").pack()
        self.buttonPause = tk.Button(self.panel_frame,text ="Pause")
        self.buttonPause.pack(anchor="s",pady=5)
        self.buttonResume = tk.Button(self.panel_frame,text ="Resume")
        self.buttonResume.pack(anchor="s",pady=5)
        self.buttonCancel = tk.Button(self.panel_frame,text ="Cancel")
        self.buttonCancel.pack(anchor="s",pady=5)

    def updateSelected(self):
        agent:Agent
        for agent in self.agents:
            if agent.selected:
                self.labelPlan.configure(text=f"Plan: {agent.plan}")
                self.labelSteps.configure(text=f"Unique Steps: {len(agent.visited)}")
                self.labelKnowledgeN.configure(text=f"Knowledge Exchanges: {agent.knowledgeTransactions}")
                # Show knowledge
                block_items = self.canvas.find_withtag("block")
                for blockID in block_items:
                    blockLoc = (int(self.canvas.gettags(blockID)[2]),int(self.canvas.gettags(blockID)[1]))
                    if blockLoc not in agent.knowledge:
                        #color = self.canvas.itemcget(blockID, "fill") #TODO GET COLOR AND SET IT TO DARKER
                        self.canvas.itemconfigure(blockID,fill="black")
                
                # Draw its route
                if len(agent.activeRoute) != 0:
                    self.drawPath(agent.activeRoute)
                return

    def createControls(self):
        def onKeyPress(event):
            if event.char == "x" and not self.started and not self.finished:
                #print("Draw Next Frame...")``
                self.moveAgentsFrame()
                self.drawNewFrame()
        def onClick(event):
            item = self.canvas.find_closest(event.x, event.y)
            tags = self.canvas.itemcget(item,"tags")
            if "Agent" in tags:
                agentName = tags.split(" ")[1]
                print(f"Clicked Agent: {agentName}")
                for agent in self.agents:
                    if(agent.name == agentName):
                        if not agent.selected:
                            agent.selected = True
                            self.canvas.itemconfigure("Agent",fill="yellow") # make all other agents yellow
                            self.canvas.itemconfigure(item, fill="blue")
                            self.labelSelected.configure(text=agent.name)
                            print(f"Plan: {agent.plan}")
                        else:
                            agent.selected = False
                            self.canvas.itemconfigure("Agent",fill="yellow") # make all other agents yellow
                            self.labelSelected.configure(text="None")
                            self.labelPlan.configure(text=f"Plan: None")
                        #self.drawNewFrame()
                    else:
                        agent.selected = False
                self.drawNewFrame()
        def hover(event):
            item = self.canvas.find_closest(event.x, event.y)
            tags = self.canvas.itemcget(item,"tags")
            if "block" in tags:
                tags = tags.split(" ")
                self.labelCords.configure(text=f"BlockCords: (x={tags[2]},y={tags[1]})")

        # Canvas controls
        self.canvas.bind("<Motion>",hover)
        self.bind("<KeyPress>",onKeyPress)
        self.bind("<Button-1>",onClick)

        def pause(_):
            self.started = False
        def resume(_):
            if self.finished:return
            self.started = True
        def cancel(_):
            self.destroy()
        
        # Buttons
        self.buttonPause.bind("<Button>",pause)
        self.buttonResume.bind("<Button>",resume)
        self.buttonCancel.bind("<Button>",cancel)
    
    def create_city(self,mapFile:str,planFile:str,agentN:int)->City:
        try:
            with open(mapFile,"r") as f:
                lines = [line.strip('\n') for line in f.readlines()] # remove \n
                self.city = City(lines, planFile,agentN=agentN)
                self.agents = self.city.agents
        except Exception as e: # catch so i can destroy this
            self.destroy()
            raise Exception(e)

    def beginSimulation(self):
        self.elapsed = time.perf_counter()
        self.started = True
        def gameLoop():
            if self.started:
                self.moveAgentsFrame()
                self.drawNewFrame()
                self.loop = self.after(0,gameLoop)
            else:
                self.update()
                self.loop = self.after(0,gameLoop)

        self.drawNewFrame()
        self.loop = self.after(0,gameLoop)
        self.mainloop()

    def stopSimulation(self):
        self.elapsed = time.perf_counter() - self.elapsed
        self.started = False
        self.finished = True
        self.buttonPause.configure(state="disabled")
        self.buttonResume.configure(state="disabled")

        print("Statistics: ")
        print("Simulation Time: " + str(self.elapsed) + " seconds")
        print("Agent Data:")
        for agent in self.agents:
            print(agent.name + "Blocks Visited: " + str(len(agent.visited)) + "| Knowledge Exchanges: " + str(agent.knowledgeTransactions))

        with open("stats.txt","w") as f:
            f.write("Statistics: \nSimulation Time: " + str(self.elapsed) + " seconds\nAgent Data:\n")
            for agent in self.agents:
                f.write(agent.name + "Blocks Visited: " + str(len(agent.visited)) + "| Knowledge Exchanges: " + str(agent.knowledgeTransactions) + "\n")


    def drawNewFrame(self):
        self.canvas.delete("all")
        agentWidth = int(self.blockSize / 2)
        for y,row in enumerate(self.city.grid):
            for x,block in enumerate(row):
                x1 = x * self.blockSize
                y1 = y * self.blockSize
                x2 = x1 + self.blockSize
                y2 = y1 + self.blockSize
                if block == "*":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="gray", tags=("block",str(y),str(x)))
                elif block == "W":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="yellow", tags=("block",str(y),str(x)))
                elif block == "A":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="red", tags=("block",str(y),str(x)))
                elif block == "T":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="blue", tags=("block",str(y),str(x)))
                elif block == "G":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="green", tags=("block",str(y),str(x)))
                elif block in [str(i) for i in range(10)]:
                    self.canvas.create_polygon(x1 + self.blockSize/2,y1, x2,y2, x1,y2, fill="red", outline="black", tags=("block",str(y),str(x)))
                    self.canvas.create_text(x1 + self.blockSize/2, y1 + self.blockSize/2, text=block)
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white",outline="lightgray", outlineoffset="center", tags=("block",str(y),str(x)))
                    self.canvas.create_text(x1 + self.blockSize/2, y1 + self.blockSize/2, text=block)

        # Draw agents
        agent:Agent
        for j, agent in enumerate(self.agents):
            x, y = agent.loc
            x1 = x * self.blockSize
            y1 = y * self.blockSize
            x2 = x1 + self.blockSize
            y2 = y1 + self.blockSize

            if not agent.selected:
                if len(agent.plan) != 0:
                    self.canvas.create_oval((x1+x2)/2 - agentWidth/2 , (y1+y2)/2 - agentWidth/2, (x1+x2)/2 + agentWidth/2 , (y1+y2)/2 + agentWidth/2, fill="yellow", activewidth=2, tags=("Agent", agent.name))
                else:
                    self.canvas.create_oval((x1+x2)/2 - agentWidth/2 , (y1+y2)/2 - agentWidth/2, (x1+x2)/2 + agentWidth/2 , (y1+y2)/2 + agentWidth/2, fill="red",activewidth=2, tags=("Agent", agent.name))
            else:
                self.canvas.create_oval((x1+x2)/2 - agentWidth/2 , (y1+y2)/2 - agentWidth/2, (x1+x2)/2 + agentWidth/2 , (y1+y2)/2 + agentWidth/2, fill="blue", activewidth=2, tags=("Agent", agent.name))
            #self.canvas.create_text(x1 + self.blockSize/2, y1 + self.blockSize/2, text=agent.house, tags=("Agent", agent.name))

        self.updateSelected()
        self.canvas.update()

    def moveAgentsFrame(self):
        agent:Agent
        for agent in self.agents:
            # Get 4 possible moves (Left, Right, Up, Down)
            possibleMoves = [
                [agent.loc[0] + 1, agent.loc[1]],
                [agent.loc[0] - 1, agent.loc[1]],
                [agent.loc[0], agent.loc[1] + 1],
                [agent.loc[0], agent.loc[1] - 1]
                ]
            
            # Filters
            possibleMoves = list(filter(lambda loc: (loc[0] not in [-1,len(self.city.grid[0])]) and (loc[1] not in [-1,len(self.city.grid[0])]), possibleMoves)) # Filter out of bounds
            possibleMoves = list(filter(lambda loc: self.city.grid[loc[1]][loc[0]] not in buildings, possibleMoves)) # Filter out Buildings
            preferredMoves = list(filter(lambda loc: tuple(loc) not in agent.visited,possibleMoves)) # Remove already visited

            # path = self.astar((0,0),(9,13),self.city.getBuildings(),[(x,y)for x in range(20) for y in range(20)])
            # self.drawPath(path)
            # Choose Move

            if len(agent.activeRoute) != 0:
                chosenMove = agent.activeRoute.pop(0)
            else:
                chosenMove = random.choice(preferredMoves if len(preferredMoves) != 0 else possibleMoves) # Prefer not visited blocks

            # Get vision of next move and add to knowledge
            nearbyBlocks = [
                [chosenMove[0] + 1, chosenMove[1]],
                [chosenMove[0] - 1, chosenMove[1]],
                [chosenMove[0], chosenMove[1] + 1],
                [chosenMove[0], chosenMove[1] - 1]
                ]
            
            # Filter vision
            nearbyBlocks = list(filter(lambda loc: (loc[0] not in [-1,len(self.city.grid[0])]) and (loc[1] not in [-1,len(self.city.grid[0])]), nearbyBlocks)) # Filter out of bounds
            agent.move(chosenMove) # Move Agent
            agent.updateKnowledge(nearbyBlocks)  # Update Knowledge

            # Check for simulation end
            if(len(agent.plan) == 0 and tuple(agent.loc) == agent.house[1]):
                self.stopSimulation()
                return

        for agent in self.agents:

            # Get nearby blocks
            nearbyBlocks =[
                self.city.grid[agent.loc[1]][agent.loc[0] + 1] if agent.loc[0] + 1 not in [-1,len(self.city.grid[0])] else None,
                self.city.grid[agent.loc[1]][agent.loc[0] - 1] if agent.loc[0] - 1 not in [-1,len(self.city.grid[0])] else None,
                self.city.grid[agent.loc[1] + 1][agent.loc[0]] if agent.loc[1] + 1 not in [-1,len(self.city.grid[0])] else None,
                self.city.grid[agent.loc[1] - 1][agent.loc[0]] if agent.loc[1] - 1 not in [-1,len(self.city.grid[0])] else None
            ]
            # Filter out None
            nearbyBlocks = list(filter(lambda block: block != None,nearbyBlocks))

            # Get nearby agents
            nearbyAgents = []
            for ag in self.agents:
                if(ag.name == agent.name): continue
                if(agent.loc[0] in [ag.loc[0]+1,ag.loc[0]-1] and agent.loc[1] == ag.loc[1]):
                    nearbyAgents.append(ag)
                elif(agent.loc[1] in [ag.loc[1]+1,ag.loc[1]-1] and agent.loc[0] == ag.loc[0]):
                    nearbyAgents.append(ag)
            agent.lookAround(nearbyBlocks,nearbyAgents)

class MainMenu(tk.Tk):
    def __init__(self, screenName: str = None, baseName: str = None, className: str = "Tk", useTk: bool = True, sync: bool = False, use: str = None) -> None:
        super().__init__(screenName, baseName, className, useTk, sync, use)
        self.title("Main Menu")
        self.geometry("400x400")
        self.mapFile = None
        self.planFile = None
        self.buildInterface()

    def buildInterface(self):
        mapButton = tk.Button(self,text ="Select Map File")
        mapButton.bind("<Button>",self.getMapFile)
        mapButton.pack(pady = 10)

        self.labelMapFile = tk.Label(self, text ="None")
        self.labelMapFile.pack()

        planButton = tk.Button(self,text ="Select Agent File")
        planButton.bind("<Button>",self.getAgentPlans)
        planButton.pack(pady = 10)

        self.labelAgentPlans = tk.Label(self, text ="None")
        self.labelAgentPlans.pack()

        lb = tk.Label(self,text="Agent Number: (>0)")
        lb.pack()
        def validation(char):
            if char.isdigit():
                return True
            else:
                self.bell()
                return False
        reg = self.register(validation)
        self.agentsEntry = tk.Entry(self,validate="key",validatecommand=(reg,"%S"))
        self.agentsEntry.insert(0,'4')
        self.agentsEntry.pack()

        self.buttonStart = tk.Button(self,text ="Start Simulation", state="disabled")
        self.buttonStart.bind("<Button>",self.startSimulation)
        self.buttonStart.pack(pady = 10)

    def startSimulation(self,_):
        if self.mapFile == None or self.planFile == None: return
        if len(self.agentsEntry.get()) == 0: return
        try:
            simulation = Application(master = self,mapFile = self.mapFile, planFile = self.planFile, agentsN= int(self.agentsEntry.get()))
        except Exception as e:
            print(e)

    def getAgentPlans(self,_):
        filetypes = (('json files', '*.json'),('json files', '*.json'))
        filename = fd.askopenfilename(
            title="Select Agent Plans File",
            filetypes=filetypes
        )
        self.labelAgentPlans.configure(text=filename)
        print(f"Map File Selected: {filename}")
        self.planFile = filename
        if(self.mapFile != None): self.buttonStart.configure(state="active")
    
    def getMapFile(self,_):
        filetypes = (('text files', '*.txt'),('text files', '*.txt'))
        filename = fd.askopenfilename(
            title="Select Map File",
            filetypes=filetypes
        )
        self.labelMapFile.configure(text=filename)
        print(f"Map File Selected: {filename}")
        self.mapFile = filename
        if(self.planFile != None): self.buttonStart.configure(state="active")

def main()->None:
    app = MainMenu()
    app.mainloop()

if(__name__ =="__main__"):
    main()